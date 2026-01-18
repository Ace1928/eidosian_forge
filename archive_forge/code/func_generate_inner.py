from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
def generate_inner(self, schema: CoreSchemaOrField) -> JsonSchemaValue:
    """Generates a JSON schema for a given core schema.

        Args:
            schema: The given core schema.

        Returns:
            The generated JSON schema.
        """
    if 'ref' in schema:
        core_ref = CoreRef(schema['ref'])
        core_mode_ref = (core_ref, self.mode)
        if core_mode_ref in self.core_to_defs_refs and self.core_to_defs_refs[core_mode_ref] in self.definitions:
            return {'$ref': self.core_to_json_refs[core_mode_ref]}
    metadata_handler = _core_metadata.CoreMetadataHandler(schema)

    def populate_defs(core_schema: CoreSchema, json_schema: JsonSchemaValue) -> JsonSchemaValue:
        if 'ref' in core_schema:
            core_ref = CoreRef(core_schema['ref'])
            defs_ref, ref_json_schema = self.get_cache_defs_ref_schema(core_ref)
            json_ref = JsonRef(ref_json_schema['$ref'])
            self.json_to_defs_refs[json_ref] = defs_ref
            if json_schema.get('$ref', None) != json_ref:
                self.definitions[defs_ref] = json_schema
                self._core_defs_invalid_for_json_schema.pop(defs_ref, None)
            json_schema = ref_json_schema
        return json_schema

    def convert_to_all_of(json_schema: JsonSchemaValue) -> JsonSchemaValue:
        if '$ref' in json_schema and len(json_schema.keys()) > 1:
            json_schema = json_schema.copy()
            ref = json_schema.pop('$ref')
            json_schema = {'allOf': [{'$ref': ref}], **json_schema}
        return json_schema

    def handler_func(schema_or_field: CoreSchemaOrField) -> JsonSchemaValue:
        """Generate a JSON schema based on the input schema.

            Args:
                schema_or_field: The core schema to generate a JSON schema from.

            Returns:
                The generated JSON schema.

            Raises:
                TypeError: If an unexpected schema type is encountered.
            """
        json_schema: JsonSchemaValue | None = None
        if self.mode == 'serialization' and 'serialization' in schema_or_field:
            ser_schema = schema_or_field['serialization']
            json_schema = self.ser_schema(ser_schema)
        if json_schema is None:
            if _core_utils.is_core_schema(schema_or_field) or _core_utils.is_core_schema_field(schema_or_field):
                generate_for_schema_type = self._schema_type_to_method[schema_or_field['type']]
                json_schema = generate_for_schema_type(schema_or_field)
            else:
                raise TypeError(f'Unexpected schema type: schema={schema_or_field}')
        if _core_utils.is_core_schema(schema_or_field):
            json_schema = populate_defs(schema_or_field, json_schema)
            json_schema = convert_to_all_of(json_schema)
        return json_schema
    current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, handler_func)
    for js_modify_function in metadata_handler.metadata.get('pydantic_js_functions', ()):

        def new_handler_func(schema_or_field: CoreSchemaOrField, current_handler: GetJsonSchemaHandler=current_handler, js_modify_function: GetJsonSchemaFunction=js_modify_function) -> JsonSchemaValue:
            json_schema = js_modify_function(schema_or_field, current_handler)
            if _core_utils.is_core_schema(schema_or_field):
                json_schema = populate_defs(schema_or_field, json_schema)
            original_schema = current_handler.resolve_ref_schema(json_schema)
            ref = json_schema.pop('$ref', None)
            if ref and json_schema:
                original_schema.update(json_schema)
            return original_schema
        current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, new_handler_func)
    for js_modify_function in metadata_handler.metadata.get('pydantic_js_annotation_functions', ()):

        def new_handler_func(schema_or_field: CoreSchemaOrField, current_handler: GetJsonSchemaHandler=current_handler, js_modify_function: GetJsonSchemaFunction=js_modify_function) -> JsonSchemaValue:
            json_schema = js_modify_function(schema_or_field, current_handler)
            if _core_utils.is_core_schema(schema_or_field):
                json_schema = populate_defs(schema_or_field, json_schema)
                json_schema = convert_to_all_of(json_schema)
            return json_schema
        current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, new_handler_func)
    json_schema = current_handler(schema)
    if _core_utils.is_core_schema(schema):
        json_schema = populate_defs(schema, json_schema)
        json_schema = convert_to_all_of(json_schema)
    return json_schema