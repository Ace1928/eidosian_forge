from __future__ import annotations as _annotations
from functools import partial, partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from warnings import warn
from typing_extensions import Literal, Protocol, TypeAlias
from .._internal import _decorators, _decorators_v1
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
def root_validator(*__args, pre: bool=False, skip_on_failure: bool=False, allow_reuse: bool=False) -> Any:
    """Decorate methods on a model indicating that they should be used to validate (and perhaps
    modify) data either before or after standard model parsing/validation is performed.

    Args:
        pre (bool, optional): Whether this validator should be called before the standard
            validators (else after). Defaults to False.
        skip_on_failure (bool, optional): Whether to stop validation and return as soon as a
            failure is encountered. Defaults to False.
        allow_reuse (bool, optional): Whether to track and raise an error if another validator
            refers to the decorated function. Defaults to False.

    Returns:
        Any: A decorator that can be used to decorate a function to be used as a root_validator.
    """
    warn('Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details', DeprecationWarning, stacklevel=2)
    if __args:
        return root_validator()(*__args)
    if allow_reuse is True:
        warn(_ALLOW_REUSE_WARNING_MESSAGE, DeprecationWarning)
    mode: Literal['before', 'after'] = 'before' if pre is True else 'after'
    if pre is False and skip_on_failure is not True:
        raise PydanticUserError('If you use `@root_validator` with pre=False (the default) you MUST specify `skip_on_failure=True`. Note that `@root_validator` is deprecated and should be replaced with `@model_validator`.', code='root-validator-pre-skip')
    wrap = partial(_decorators_v1.make_v1_generic_root_validator, pre=pre)

    def dec(f: Callable[..., Any] | classmethod[Any, Any, Any] | staticmethod[Any, Any]) -> Any:
        if _decorators.is_instance_method_from_sig(f):
            raise TypeError('`@root_validator` cannot be applied to instance methods')
        res = _decorators.ensure_classmethod_based_on_signature(f)
        dec_info = _decorators.RootValidatorDecoratorInfo(mode=mode)
        return _decorators.PydanticDescriptorProxy(res, dec_info, shim=wrap)
    return dec