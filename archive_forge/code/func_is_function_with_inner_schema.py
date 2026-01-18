from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def is_function_with_inner_schema(schema: CoreSchemaOrField) -> TypeGuard[FunctionSchemaWithInnerSchema]:
    return schema['type'] in _FUNCTION_WITH_INNER_SCHEMA_TYPES