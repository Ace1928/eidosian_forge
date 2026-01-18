import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def match_differentiability_info(native_functions: List[NativeFunction], differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]]) -> List[NativeFunctionWithDifferentiabilityInfo]:
    """Sets the "derivative" key on declarations to matching autograd function
    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """
    functional_info_by_signature = {schema.signature(strip_default=True): info_dict for schema, info_dict in differentiability_infos.items() if schema.kind() == SchemaKind.functional}
    non_functional_info_by_signature = {schema.signature(strip_default=True): info_dict for schema, info_dict in differentiability_infos.items() if schema.kind() != SchemaKind.functional}

    def find_info(f: NativeFunction) -> Tuple[Optional[Dict[str, DifferentiabilityInfo]], bool]:
        if 'generated' in f.tags and f.func.kind() == SchemaKind.out:
            return (None, False)
        if f.func in differentiability_infos:
            return (differentiability_infos[f.func], True)
        f_sig = f.func.signature(strip_default=True)
        if f_sig in functional_info_by_signature and (not is_foreach_func(f)):
            return (functional_info_by_signature[f_sig], False)
        if 'generated' in f.tags and f_sig in non_functional_info_by_signature:
            info_dict = non_functional_info_by_signature[f_sig]
            assert not any((any(('self' in str(inpt.nctype.name) for inpt in info.all_saved_inputs)) for info in info_dict.values())), f'''Attempted to convert a derivative formula for a mutable operator\n to be used by automatically by its functional variant ("{str(f.func)}").\n this is not currently supported (we'd need to fix up the formula in the codegen).'''
            return (info_dict, False)
        if is_foreach_func(f):
            assert f.func not in differentiability_infos
            diff_info, is_generated = gen_foreach_derivativeinfo(f, functional_info_by_signature, non_functional_info_by_signature)
            if diff_info is None:
                return (None, False)
            diff_info_dict = {'Default': diff_info}
            if is_generated:
                differentiability_infos[f.func] = diff_info_dict
                functional_info_by_signature[f.func] = diff_info_dict
            return (diff_info_dict, is_generated)
        return (None, False)
    result: List[NativeFunctionWithDifferentiabilityInfo] = []
    for f in native_functions:
        info_dict, is_exact_match = find_info(f)
        if f.func.kind() == SchemaKind.inplace and info_dict is not None:
            for info in info_dict.values():
                for derivative in info.derivatives:
                    if 'self' in derivative.var_names:
                        for saved_input in derivative.saved_inputs:
                            assert 'strides_or_error' not in saved_input.expr, f"Calling '.strides()' in the 'self' derivative formula of an in-place function is not supported: {f.func}"
        if not info_dict:
            result.append(NativeFunctionWithDifferentiabilityInfo(func=f, info=None, fw_derivatives=None))
            continue
        fw_derivative_dict: Dict[str, Sequence[ForwardDerivative]] = {}
        for key, info in info_dict.items():
            if not info.forward_derivatives:
                fw_derivative_dict[key] = []
                continue
            forward_derivatives = info.forward_derivatives
            if f.func.kind() == SchemaKind.inplace:
                assert len(info.forward_derivatives) == 1
                fw_info = info.forward_derivatives[0]
                formula = fw_info.formula

                def replace_self_with_original_self(formula: str, postfix: str) -> str:

                    def repl(m: Match[str]) -> str:
                        return f'{m.group(1)}original_self{postfix}{m.group(2)}'
                    return re.sub(IDENT_REGEX.format(f'self{postfix}'), repl, formula)
                if re.search(IDENT_REGEX.format('self_p'), formula):
                    if is_exact_match:
                        raise RuntimeError(f'The formula for "{f.func.name}" is using the original value of self that is being modified inplace. This would lead to wrong forward gradients. Please use "result" in the formula only.')
                    else:
                        formula = replace_self_with_original_self(formula, '_p')
                        formula = replace_self_with_original_self(formula, '_t')

                def repl(m: Match[str]) -> str:
                    return f'{m.group(1)}self_p{m.group(2)}'
                formula = re.sub(IDENT_REGEX.format('result'), repl, formula)
                required_primals = fw_info.required_inputs_primal
                if re.search(IDENT_REGEX.format('self_p'), formula):
                    required_primals = required_primals + ('self',) if required_primals else ('self',)
                if not is_exact_match:
                    is_single_method_on_self_t = False
                    directly_do_inplace = False
                    op_name: Optional[str] = None
                    between_parens: Optional[str] = None
                    match = re.fullmatch('self_t.([\\w]*)\\((.*)\\)', formula)
                    if match:
                        op_name, between_parens = (match.group(1), match.group(2))

                        def check_parens_nest_level_gt_zero(s: str) -> bool:
                            level = 1
                            for ch in s:
                                if ch == ')':
                                    level -= 1
                                    if level == 0:
                                        return False
                                if ch == '(':
                                    level += 1
                            return True
                        is_single_method_on_self_t = check_parens_nest_level_gt_zero(between_parens)
                        directly_do_inplace = is_single_method_on_self_t and op_name == info.name
                    if directly_do_inplace:
                        assert op_name is not None
                        assert between_parens is not None
                        formula = f'self_t_raw.defined() ? self_t_raw.{op_name}_({between_parens}) : {formula}'
                    else:
                        formula = f'self_t_raw.defined() ? self_t_raw.copy_({formula}) : {formula}'
                required_original_self_value = bool(re.search(IDENT_REGEX.format('original_self_p'), formula)) or bool(re.search(IDENT_REGEX.format('original_self_t'), formula))
                forward_derivatives = [ForwardDerivative(formula=formula, var_names=('self',), var_types=fw_info.var_types, required_inputs_fw_grad=fw_info.required_inputs_fw_grad, required_inputs_primal=required_primals, required_original_self_value=required_original_self_value, is_reusing_outplace_formula=not is_exact_match)]
            fw_derivative_dict[key] = forward_derivatives
        result.append(NativeFunctionWithDifferentiabilityInfo(func=f, info=info_dict, fw_derivatives=fw_derivative_dict))
    return result