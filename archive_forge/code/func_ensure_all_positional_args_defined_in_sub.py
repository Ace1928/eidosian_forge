import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def ensure_all_positional_args_defined_in_sub(super_sig: inspect.Signature, sub_sig: inspect.Signature, super_type_hints: Dict, sub_type_hints: Dict, check_first_parameter: bool, is_same_main_module: bool, method_name: str):
    sub_parameter_values = [v for v in sub_sig.parameters.values() if v.kind not in (Parameter.KEYWORD_ONLY, Parameter.VAR_KEYWORD)]
    super_parameter_values = [v for v in super_sig.parameters.values() if v.kind not in (Parameter.KEYWORD_ONLY, Parameter.VAR_KEYWORD)]
    sub_has_var_args = any((p.kind == Parameter.VAR_POSITIONAL for p in sub_parameter_values))
    super_has_var_args = any((p.kind == Parameter.VAR_POSITIONAL for p in super_parameter_values))
    if not sub_has_var_args and len(sub_parameter_values) < len(super_parameter_values):
        raise TypeError(f'{method_name}: parameter list too short')
    super_shift = 0
    for index, sub_param in enumerate(sub_parameter_values):
        if index == 0 and (not check_first_parameter):
            continue
        if index + super_shift >= len(super_parameter_values):
            if sub_param.kind == Parameter.VAR_POSITIONAL:
                continue
            if sub_param.kind == Parameter.POSITIONAL_ONLY and sub_param.default != Parameter.empty:
                continue
            if sub_param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                continue
            raise TypeError(f'{method_name}: `{sub_param.name}` positionally required in subclass but not in supertype')
        if sub_param.kind == Parameter.VAR_POSITIONAL:
            return
        super_param = super_parameter_values[index + super_shift]
        if super_param.kind == Parameter.VAR_POSITIONAL:
            super_shift -= 1
        if super_param.kind == Parameter.VAR_POSITIONAL:
            if not sub_has_var_args:
                raise TypeError(f'{method_name}: `{super_param.name}` must be present')
            continue
        if super_param.kind != sub_param.kind and (not (super_param.kind == Parameter.POSITIONAL_ONLY and sub_param.kind == Parameter.POSITIONAL_OR_KEYWORD)) and (not (sub_param.kind == Parameter.POSITIONAL_ONLY and super_has_var_args)):
            raise TypeError(f'{method_name}: `{sub_param.name}` is not `{super_param.kind}` and is `{sub_param.kind}`')
        elif (super_param.name in super_type_hints or is_same_main_module) and (not _issubtype(super_type_hints.get(super_param.name, None), sub_type_hints.get(sub_param.name, None))):
            raise TypeError(f'`{method_name}: {sub_param.name} overriding must be a supertype of `{super_param.annotation}` but is `{sub_param.annotation}`')