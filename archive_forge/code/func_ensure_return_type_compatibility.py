import inspect
from inspect import Parameter
from types import FunctionType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
from .typing_utils import get_args, issubtype
def ensure_return_type_compatibility(super_type_hints: Dict, sub_type_hints: Dict, method_name: str):
    super_return = super_type_hints.get('return', None)
    sub_return = sub_type_hints.get('return', None)
    if not _issubtype(sub_return, super_return) and super_return is not None:
        raise TypeError(f'{method_name}: return type `{sub_return}` is not a `{super_return}`.')