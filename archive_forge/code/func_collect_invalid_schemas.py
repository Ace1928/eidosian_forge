from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def collect_invalid_schemas(schema: core_schema.CoreSchema) -> bool:
    invalid = False

    def _is_schema_valid(s: core_schema.CoreSchema, recurse: Recurse) -> core_schema.CoreSchema:
        nonlocal invalid
        if 'metadata' in s:
            metadata = s['metadata']
            if HAS_INVALID_SCHEMAS_METADATA_KEY in metadata:
                invalid = metadata[HAS_INVALID_SCHEMAS_METADATA_KEY]
                return s
        return recurse(s, _is_schema_valid)
    walk_core_schema(schema, _is_schema_valid)
    return invalid