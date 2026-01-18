import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def ensure_all_kwargs_defined_in_sub(super_sig: inspect.Signature, sub_sig: inspect.Signature, super_type_hints: Dict, sub_type_hints: Dict, check_first_parameter: bool, method_name: str):
    sub_has_var_kwargs = any((p.kind == Parameter.VAR_KEYWORD for p in sub_sig.parameters.values()))
    for super_index, (name, super_param) in enumerate(super_sig.parameters.items()):
        if super_index == 0 and (not check_first_parameter):
            continue
        if super_param.kind == Parameter.VAR_POSITIONAL:
            continue
        if super_param.kind == Parameter.POSITIONAL_ONLY:
            continue
        if not is_param_defined_in_sub(name, True, sub_has_var_kwargs, sub_sig, super_param):
            raise TypeError(f'{method_name}: `{name}` is not present.')
        elif name in sub_sig.parameters and super_param.kind != Parameter.VAR_KEYWORD:
            sub_index = list(sub_sig.parameters.keys()).index(name)
            sub_param = sub_sig.parameters[name]
            if super_param.kind != sub_param.kind and (not (super_param.kind == Parameter.KEYWORD_ONLY and sub_param.kind == Parameter.POSITIONAL_OR_KEYWORD)):
                raise TypeError(f'{method_name}: `{name}` is not `{super_param.kind}`')
            elif super_index > sub_index and super_param.kind != Parameter.KEYWORD_ONLY:
                raise TypeError(f'{method_name}: `{name}` is not parameter at index `{super_index}`')
            elif name in super_type_hints and name in sub_type_hints and (not _issubtype(super_type_hints[name], sub_type_hints[name])):
                raise TypeError(f'`{method_name}: {name} must be a supertype of `{super_param.annotation}` but is `{sub_param.annotation}`')