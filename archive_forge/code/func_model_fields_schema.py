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
def model_fields_schema(self, schema: core_schema.ModelFieldsSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a model's fields.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    named_required_fields: list[tuple[str, bool, CoreSchemaField]] = [(name, self.field_is_required(field, total=True), field) for name, field in schema['fields'].items() if self.field_is_present(field)]
    if self.mode == 'serialization':
        named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
    json_schema = self._named_required_fields_schema(named_required_fields)
    extras_schema = schema.get('extras_schema', None)
    if extras_schema is not None:
        schema_to_update = self.resolve_schema_to_update(json_schema)
        schema_to_update['additionalProperties'] = self.generate_inner(extras_schema)
    return json_schema