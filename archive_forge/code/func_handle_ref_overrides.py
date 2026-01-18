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
def handle_ref_overrides(self, json_schema: JsonSchemaValue) -> JsonSchemaValue:
    """It is not valid for a schema with a top-level $ref to have sibling keys.

        During our own schema generation, we treat sibling keys as overrides to the referenced schema,
        but this is not how the official JSON schema spec works.

        Because of this, we first remove any sibling keys that are redundant with the referenced schema, then if
        any remain, we transform the schema from a top-level '$ref' to use allOf to move the $ref out of the top level.
        (See bottom of https://swagger.io/docs/specification/using-ref/ for a reference about this behavior)
        """
    if '$ref' in json_schema:
        json_schema = json_schema.copy()
        referenced_json_schema = self.get_schema_from_definitions(JsonRef(json_schema['$ref']))
        if referenced_json_schema is None:
            if len(json_schema) > 1:
                json_schema = json_schema.copy()
                json_schema.setdefault('allOf', [])
                json_schema['allOf'].append({'$ref': json_schema['$ref']})
                del json_schema['$ref']
            return json_schema
        for k, v in list(json_schema.items()):
            if k == '$ref':
                continue
            if k in referenced_json_schema and referenced_json_schema[k] == v:
                del json_schema[k]
        if len(json_schema) > 1:
            json_ref = JsonRef(json_schema['$ref'])
            del json_schema['$ref']
            assert 'allOf' not in json_schema
            json_schema['allOf'] = [{'$ref': json_ref}]
    return json_schema