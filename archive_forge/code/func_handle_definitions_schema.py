from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def handle_definitions_schema(self, schema: core_schema.DefinitionsSchema, f: Walk) -> core_schema.CoreSchema:
    new_definitions: list[core_schema.CoreSchema] = []
    for definition in schema['definitions']:
        if 'schema_ref' in definition and 'ref' in definition:
            new_definitions.append(definition)
            self.walk(definition, f)
            continue
        updated_definition = self.walk(definition, f)
        if 'ref' in updated_definition:
            new_definitions.append(updated_definition)
    new_inner_schema = self.walk(schema['schema'], f)
    if not new_definitions and len(schema) == 3:
        return new_inner_schema
    new_schema = schema.copy()
    new_schema['schema'] = new_inner_schema
    new_schema['definitions'] = new_definitions
    return new_schema