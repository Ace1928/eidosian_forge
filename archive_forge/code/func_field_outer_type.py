from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union, Generic, TypeVar, Callable, cast, overload
from datetime import date, datetime
from typing_extensions import Self
import pydantic
from pydantic.fields import FieldInfo
from ._types import StrBytesIntFloat
def field_outer_type(field: FieldInfo) -> Any:
    if PYDANTIC_V2:
        return field.annotation
    return field.outer_type_