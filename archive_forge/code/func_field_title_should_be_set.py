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
def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
    """Returns true if a field with the given schema should have a title set based on the field name.

        Intuitively, we want this to return true for schemas that wouldn't otherwise provide their own title
        (e.g., int, float, str), and false for those that would (e.g., BaseModel subclasses).

        Args:
            schema: The schema to check.

        Returns:
            `True` if the field should have a title set, `False` otherwise.
        """
    if _core_utils.is_core_schema_field(schema):
        if schema['type'] == 'computed-field':
            field_schema = schema['return_schema']
        else:
            field_schema = schema['schema']
        return self.field_title_should_be_set(field_schema)
    elif _core_utils.is_core_schema(schema):
        if schema.get('ref'):
            return False
        if schema['type'] in {'default', 'nullable', 'definitions'}:
            return self.field_title_should_be_set(schema['schema'])
        if _core_utils.is_function_with_inner_schema(schema):
            return self.field_title_should_be_set(schema['schema'])
        if schema['type'] == 'definition-ref':
            return False
        return True
    else:
        raise PydanticInvalidForJsonSchema(f'Unexpected schema type: schema={schema}')