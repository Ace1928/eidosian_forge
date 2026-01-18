import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
def postprocess_forward_derivatives(f: NativeFunction, defn_name: str, all_arg_names: List[str], derivatives: List[Derivative], forward_derivatives: List[ForwardDerivative], args_with_derivatives: Sequence[Binding]) -> List[ForwardDerivative]:

    def find_required_inputs(formula: str, postfix: str) -> Tuple[str, ...]:
        is_foreach = f.func.name.name.base.startswith('_foreach_')
        required_inputs = set()
        for arg in args_with_derivatives:
            if arg.type in ('at::TensorList', 'const at::ITensorListRef &') and (not is_foreach):
                continue
            arg_name = arg.name
            found = re.search(IDENT_REGEX.format(arg_name), formula)
            if found:
                raise RuntimeError(f'The forward formula for {defn_name} is using the base name of the {arg_name} argument which is ambiguous. You should use {arg_name}_p to access the primal value and {arg_name}_t to access the tangent.')
            found = re.search(IDENT_REGEX.format(arg_name + postfix), formula)
            if found:
                required_inputs.add(arg_name)
        return tuple(required_inputs)
    updated_derivatives: List[ForwardDerivative] = []
    for defn in forward_derivatives:
        formula = defn.formula
        required_inputs_tangent = find_required_inputs(formula, '_t')
        if formula == 'auto_element_wise':
            assert f.func.kind() != SchemaKind.inplace, f'Cannot use auto_element_wise with {f.func.name} because it is an in-place variant'
            if not len(args_with_derivatives) == 1 or len(forward_derivatives) > 1 or len(forward_derivatives[0].var_names) > 1:
                raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml defines the forward definition of gradient as element_wise but this only works for functions with a single differentiable input and a single differentiable output.')
            if not len(derivatives) == 1:
                raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml defines the forward definition of gradient as element_wise but it does not defines the gradient formula for its argument which is required.')
            backward_formula = derivatives[0].original_formula
            input_name = args_with_derivatives[0].name

            def repl(m: Any) -> str:
                return f'{m.group(1)}{input_name}_t.conj(){m.group(2)}'
            fw_formula = re.sub(IDENT_REGEX.format('grad'), repl, backward_formula)
            for arg in args_with_derivatives:
                arg_name = arg.name

                def repl(m: Any) -> str:
                    return f'{m.group(1)}{arg_name}_p{m.group(2)}'
                fw_formula = re.sub(IDENT_REGEX.format(arg_name), repl, fw_formula)
            fw_formula = f'({fw_formula}).conj()'
            required_inputs_tangent = tuple(all_arg_names)
            formula = fw_formula
        elif formula == 'auto_linear':
            if len(forward_derivatives) > 1 or len(forward_derivatives[0].var_names) > 1:
                raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml defines the forward definition of gradient as linear but this only works for functions with a single differentiable output.')
            diff_arg_names = [arg.name for arg in args_with_derivatives]
            assert len(diff_arg_names) > 0
            new_args = []
            for arg_name in all_arg_names:
                if arg_name in diff_arg_names:
                    arg_name = arg_name + '_t'
                new_args.append(arg_name)
            if f.func.has_symint():
                defn_name += '_symint'
            if Variant.function in f.variants:
                fw_formula = f'at::{defn_name}({', '.join(new_args)})'
            else:
                assert Variant.method in f.variants
                fw_formula = f'{new_args[0]}.{defn_name}({', '.join(new_args[1:])})'
            required_inputs_tangent = tuple(diff_arg_names)
            formula = fw_formula
        required_inputs_primal = find_required_inputs(formula, '_p')
        updated_derivatives.append(ForwardDerivative(formula=formula, var_names=defn.var_names, var_types=defn.var_types, required_inputs_fw_grad=required_inputs_tangent, required_inputs_primal=required_inputs_primal, required_original_self_value=False, is_reusing_outplace_formula=False))
    return updated_derivatives