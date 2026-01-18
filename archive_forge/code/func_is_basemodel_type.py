from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Any, Type, Union, Generic, TypeVar, Callable, cast
from datetime import date, datetime
from typing_extensions import (
import pydantic
import pydantic.generics
from pydantic.fields import FieldInfo
from ._types import (
from ._utils import (
from ._compat import (
from ._constants import RAW_RESPONSE_HEADER
def is_basemodel_type(type_: type) -> TypeGuard[type[BaseModel] | type[GenericModel]]:
    origin = get_origin(type_) or type_
    return issubclass(origin, BaseModel) or issubclass(origin, GenericModel)