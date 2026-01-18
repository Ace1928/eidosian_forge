import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def gen_foreach_derivativeinfo(foreach_function: NativeFunction, functional_info_by_signature: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], non_functional_info_by_signature: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], dispatch_key: str='Default') -> Tuple[Optional[DifferentiabilityInfo], bool]:
    """Generate DifferentiabilityInfo for out-place foreach function, return the existing one for in-place.

    The second return value indicates whether the info is generated in this function.
    """
    ref_diff_info: Optional[DifferentiabilityInfo] = None
    for function_schema, diff_info in functional_info_by_signature.items():
        if not is_reference_for_foreach(foreach_function, function_schema):
            continue
        ref_diff_info = diff_info[dispatch_key]
        if ref_diff_info is not None:
            break
    if ref_diff_info is None and foreach_function.func.kind() == SchemaKind.inplace and (str(foreach_function.func.name) in _foreach_with_inplace_ref):
        for function_schema, diff_info in non_functional_info_by_signature.items():
            if not is_reference_for_foreach(foreach_function, function_schema):
                continue
            ref_diff_info = diff_info[dispatch_key]
            if ref_diff_info is not None:
                break
    if ref_diff_info is None:
        return (None, False)
    if foreach_function.func.kind() == SchemaKind.inplace:
        return (ref_diff_info, False)
    map_refarg2foreacharg, map_name2arg = ({}, {})
    for i, (arg, ref_arg) in enumerate(zip(foreach_function.func.arguments.flat_non_out, function_schema.arguments.flat_non_out)):
        map_refarg2foreacharg[ref_arg.name] = arg.name
        map_name2arg[arg.name] = arg
    all_saved_inputs, all_saved_outputs, all_var_names = ([], [], [])
    modified_derivative_formulas = []
    for i, derivative in enumerate(ref_diff_info.derivatives):
        modified_formula = derivative.formula.replace('grad', 'grads[i]').replace('result', 'result[i]')
        saved_inputs, saved_outputs = ([], [])
        with local.parametrize(use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors, use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group):
            for ref_input in derivative.saved_inputs:
                ref_input_jit_name = ref_input.expr.split('.')[0]
                mapped_name = map_refarg2foreacharg[ref_input_jit_name]
                if isinstance(map_name2arg[mapped_name].type, ListType):
                    mapped_expr = mapped_name + '[i]'
                else:
                    mapped_expr = mapped_name
                new_expr = ref_input.expr.replace(ref_input_jit_name, mapped_expr)
                modified_formula = modified_formula.replace(cast(str, ref_input.nctype.name), new_expr)
                nctype = cpp.argument_type(map_name2arg[mapped_name], binds=mapped_name)
                canonical_nctype = NamedCType(nctype.name, nctype.type.remove_const_ref())
                saved_inputs.append(SavedAttribute(nctype=canonical_nctype, expr=mapped_name))
            for ref_output in derivative.saved_outputs:
                if ref_output.nctype.name == 'result':
                    saved_outputs.append(SavedAttribute(nctype=NamedCType(name='result', type=BaseCType(tensorListT)), expr='result'))
                else:
                    raise RuntimeError('')
        var_names = [map_refarg2foreacharg[var] for var in derivative.var_names]
        all_var_names.extend(var_names)
        all_saved_inputs.extend(saved_inputs)
        all_saved_outputs.extend(saved_outputs)
        modified_derivative = Derivative(formula=modified_formula, original_formula=derivative.formula, var_names=tuple(var_names), saved_inputs=tuple(saved_inputs), saved_outputs=tuple(saved_outputs), named_gradients=set())
        modified_derivative_formulas.append(modified_derivative)
    with local.parametrize(use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors, use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group):
        args_with_derivatives = [Binding(name=arg.name, nctype=cpp.argument_type(arg, binds=arg.name), argument=arg, default=None) for arg in foreach_function.func.arguments.flat_non_out if arg.name in all_var_names]
    forward_derivatives: List[ForwardDerivative] = []
    fw_derivative: ForwardDerivative
    for fw_derivative in ref_diff_info.forward_derivatives:
        var_names: List[str] = list(fw_derivative.var_names)
        var_types: List[Type] = list(fw_derivative.var_types)
        required_inputs_fw_grad: List[str] = []
        required_inputs_primal: List[str] = []
        if fw_derivative.required_inputs_fw_grad is not None:
            required_inputs_fw_grad = list(fw_derivative.required_inputs_fw_grad)
        if fw_derivative.required_inputs_primal:
            required_inputs_primal = list(fw_derivative.required_inputs_primal)
        modified_formula = fw_derivative.formula
        if 'result' in modified_formula:
            modified_formula = fw_derivative.formula.replace('result', 'result[i]')
        for foreach_arg, ref_arg in zip(foreach_function.func.arguments.flat_non_out, ref_diff_info.func.func.arguments.flat_non_out):
            if isinstance(foreach_arg.type, ListType) and (not foreach_arg.type.is_tensor_like()):
                modified_formula = modified_formula.replace(ref_arg.name, foreach_arg.name + '[i]')
            elif foreach_arg.type.is_tensor_like():
                assert isinstance(foreach_arg.type, ListType) or (foreach_arg.type == BaseType(BaseTy.Tensor) and str(foreach_function.func.name) in _foreach_with_tensor_overload), f'{foreach_function.func.name}, {foreach_arg.type}'
                for suffix in ('_p', '_t'):
                    curr_expr = ref_arg.name + suffix
                    if curr_expr in modified_formula:
                        new_expr = foreach_arg.name + suffix
                        modified_formula = modified_formula.replace(curr_expr, new_expr)
            elif foreach_arg.name != ref_arg.name:
                modified_formula = modified_formula.replace(ref_arg.name, foreach_arg.name)
            for i, name in enumerate(var_names):
                if name == ref_arg.name:
                    var_names[i] = foreach_arg.name
                    var_types[i] = foreach_arg.type
            for i, name in enumerate(required_inputs_fw_grad):
                if name == ref_arg.name:
                    required_inputs_fw_grad[i] = foreach_arg.name
            for i, name in enumerate(required_inputs_primal):
                if name == ref_arg.name:
                    required_inputs_primal[i] = foreach_arg.name
        forward_derivatives.append(ForwardDerivative(formula=modified_formula, var_names=tuple(var_names), var_types=tuple(var_types), required_inputs_fw_grad=tuple(required_inputs_fw_grad), required_inputs_primal=tuple(required_inputs_primal), required_original_self_value=fw_derivative.required_original_self_value, is_reusing_outplace_formula=fw_derivative.is_reusing_outplace_formula))
    return (DifferentiabilityInfo(name=foreach_function.func.name.name.base, func=foreach_function, op=f'Foreach{ref_diff_info.op}{foreach_function.func.name.overload_name}', derivatives=modified_derivative_formulas, forward_derivatives=forward_derivatives, all_saved_inputs=tuple(set(all_saved_inputs)), all_saved_outputs=tuple(set(all_saved_outputs)), available_named_gradients=(), used_named_gradients=set(), args_with_derivatives=args_with_derivatives, non_differentiable_arg_names=[], output_differentiability=None, output_differentiability_conditions=None), True)