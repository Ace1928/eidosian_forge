import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def ensure_no_extra_args_in_sub(super_sig: inspect.Signature, sub_sig: inspect.Signature, check_first_parameter: bool, method_name: str) -> None:
    super_params = super_sig.parameters.values()
    super_var_args = any((p.kind == Parameter.VAR_POSITIONAL for p in super_params))
    super_var_kwargs = any((p.kind == Parameter.VAR_KEYWORD for p in super_params))
    for sub_index, (name, sub_param) in enumerate(sub_sig.parameters.items()):
        if sub_param.kind == Parameter.POSITIONAL_ONLY and len(super_params) > sub_index and (list(super_params)[sub_index].kind == Parameter.POSITIONAL_ONLY):
            continue
        if name not in super_sig.parameters and sub_param.default == Parameter.empty and (sub_param.kind != Parameter.VAR_POSITIONAL) and (sub_param.kind != Parameter.VAR_KEYWORD) and (not (sub_param.kind == Parameter.KEYWORD_ONLY and super_var_kwargs)) and (not (sub_param.kind == Parameter.POSITIONAL_ONLY and super_var_args)) and (not (sub_param.kind == Parameter.POSITIONAL_OR_KEYWORD and super_var_args)) and (sub_index > 0 or check_first_parameter):
            raise TypeError(f'{method_name}: `{name}` is not a valid parameter.')