from __future__ import annotations as _annotations
import dataclasses
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Generic, NoReturn, TypeVar, overload
from typing_extensions import Literal, TypeGuard, dataclass_transform
from ._internal import _config, _decorators, _typing_extra
from ._internal import _dataclasses as _pydantic_dataclasses
from ._migration import getattr_migration
from .config import ConfigDict
from .fields import Field, FieldInfo
def make_pydantic_fields_compatible(cls: type[Any]) -> None:
    """Make sure that stdlib `dataclasses` understands `Field` kwargs like `kw_only`
        To do that, we simply change
          `x: int = pydantic.Field(..., kw_only=True)`
        into
          `x: int = dataclasses.field(default=pydantic.Field(..., kw_only=True), kw_only=True)`
        """
    for annotation_cls in cls.__mro__:
        annotations = getattr(annotation_cls, '__annotations__', [])
        for field_name in annotations:
            field_value = getattr(cls, field_name, None)
            if not isinstance(field_value, FieldInfo):
                continue
            field_args: dict = {'default': field_value}
            if sys.version_info >= (3, 10) and field_value.kw_only:
                field_args['kw_only'] = True
            if field_value.repr is not True:
                field_args['repr'] = field_value.repr
            setattr(cls, field_name, dataclasses.field(**field_args))
            if cls.__dict__.get('__annotations__') is None:
                cls.__annotations__ = {}
            cls.__annotations__[field_name] = annotations[field_name]