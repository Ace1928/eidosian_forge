from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
from pydantic_core import SchemaSerializer, SchemaValidator
from typing_extensions import Literal
from ..errors import PydanticErrorCodes, PydanticUserError
def attempt_rebuild_validator() -> SchemaValidator | None:
    if rebuild_dataclass(cls, raise_errors=False, _parent_namespace_depth=5) is not False:
        return cls.__pydantic_validator__
    else:
        return None