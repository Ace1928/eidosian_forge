from __future__ import annotations as _annotations
import dataclasses
import sys
from functools import partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, cast, overload
from pydantic_core import core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import GetCoreSchemaHandler as _GetCoreSchemaHandler
from ._internal import _core_metadata, _decorators, _generics, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
from .errors import PydanticUserError
def field_validator(__field: str, *fields: str, mode: FieldValidatorModes='after', check_fields: bool | None=None) -> Callable[[Any], Any]:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/validators/#field-validators

    Decorate methods on the class indicating that they should be used to validate fields.

    Example usage:
    ```py
    from typing import Any

    from pydantic import (
        BaseModel,
        ValidationError,
        field_validator,
    )

    class Model(BaseModel):
        a: str

        @field_validator('a')
        @classmethod
        def ensure_foobar(cls, v: Any):
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v

    print(repr(Model(a='this is foobar good')))
    #> Model(a='this is foobar good')

    try:
        Model(a='snap')
    except ValidationError as exc_info:
        print(exc_info)
        '''
        1 validation error for Model
        a
          Value error, "foobar" not found in a [type=value_error, input_value='snap', input_type=str]
        '''
    ```

    For more in depth examples, see [Field Validators](../concepts/validators.md#field-validators).

    Args:
        __field: The first field the `field_validator` should be called on; this is separate
            from `fields` to ensure an error is raised if you don't pass at least one.
        *fields: Additional field(s) the `field_validator` should be called on.
        mode: Specifies whether to validate the fields before or after validation.
        check_fields: Whether to check that the fields actually exist on the model.

    Returns:
        A decorator that can be used to decorate a function to be used as a field_validator.

    Raises:
        PydanticUserError:
            - If `@field_validator` is used bare (with no fields).
            - If the args passed to `@field_validator` as fields are not strings.
            - If `@field_validator` applied to instance methods.
    """
    if isinstance(__field, FunctionType):
        raise PydanticUserError("`@field_validator` should be used with fields and keyword arguments, not bare. E.g. usage should be `@validator('<field_name>', ...)`", code='validator-no-fields')
    fields = (__field, *fields)
    if not all((isinstance(field, str) for field in fields)):
        raise PydanticUserError("`@field_validator` fields should be passed as separate string args. E.g. usage should be `@validator('<field_name_1>', '<field_name_2>', ...)`", code='validator-invalid-fields')

    def dec(f: Callable[..., Any] | staticmethod[Any, Any] | classmethod[Any, Any, Any]) -> _decorators.PydanticDescriptorProxy[Any]:
        if _decorators.is_instance_method_from_sig(f):
            raise PydanticUserError('`@field_validator` cannot be applied to instance methods', code='validator-instance-method')
        f = _decorators.ensure_classmethod_based_on_signature(f)
        dec_info = _decorators.FieldValidatorDecoratorInfo(fields=fields, mode=mode, check_fields=check_fields)
        return _decorators.PydanticDescriptorProxy(f, dec_info)
    return dec