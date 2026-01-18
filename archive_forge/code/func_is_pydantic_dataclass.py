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
def is_pydantic_dataclass(__cls: type[Any]) -> TypeGuard[type[PydanticDataclass]]:
    """Whether a class is a pydantic dataclass.

    Args:
        __cls: The class.

    Returns:
        `True` if the class is a pydantic dataclass, `False` otherwise.
    """
    return dataclasses.is_dataclass(__cls) and '__pydantic_validator__' in __cls.__dict__