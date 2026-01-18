from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union, Generic, TypeVar, Callable, cast, overload
from datetime import date, datetime
from typing_extensions import Self
import pydantic
from pydantic.fields import FieldInfo
from ._types import StrBytesIntFloat
def field_get_default(field: FieldInfo) -> Any:
    value = field.get_default()
    if PYDANTIC_V2:
        from pydantic_core import PydanticUndefined
        if value == PydanticUndefined:
            return None
        return value
    return value