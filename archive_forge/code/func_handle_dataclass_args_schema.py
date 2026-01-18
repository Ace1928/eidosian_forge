from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def handle_dataclass_args_schema(self, schema: core_schema.DataclassArgsSchema, f: Walk) -> core_schema.CoreSchema:
    replaced_fields: list[core_schema.DataclassField] = []
    replaced_computed_fields: list[core_schema.ComputedField] = []
    for computed_field in schema.get('computed_fields', ()):
        replaced_field = computed_field.copy()
        replaced_field['return_schema'] = self.walk(computed_field['return_schema'], f)
        replaced_computed_fields.append(replaced_field)
    if replaced_computed_fields:
        schema['computed_fields'] = replaced_computed_fields
    for field in schema['fields']:
        replaced_field = field.copy()
        replaced_field['schema'] = self.walk(field['schema'], f)
        replaced_fields.append(replaced_field)
    schema['fields'] = replaced_fields
    return schema