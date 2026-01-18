from dataclasses import Field, MISSING, _FIELDS, _FIELD, _FIELD_INITVAR  # type: ignore
from typing import Type, Any, TypeVar, List
from .data import Data
from .types import is_optional
def get_default_value_for_field(field: Field) -> Any:
    if field.default != MISSING:
        return field.default
    elif field.default_factory != MISSING:
        return field.default_factory()
    elif is_optional(field.type):
        return None
    raise DefaultValueNotFoundError()