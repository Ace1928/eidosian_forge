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
from .errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, PydanticUserError
def enum_schema(self, schema: core_schema.EnumSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches an Enum value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    enum_type = schema['cls']
    description = None if not enum_type.__doc__ else inspect.cleandoc(enum_type.__doc__)
    if description == 'An enumeration.':
        description = None
    result: dict[str, Any] = {'title': enum_type.__name__, 'description': description}
    result = {k: v for k, v in result.items() if v is not None}
    expected = [to_jsonable_python(v.value) for v in schema['members']]
    result['enum'] = expected
    if len(expected) == 1:
        result['const'] = expected[0]
    types = {type(e) for e in expected}
    if isinstance(enum_type, str) or types == {str}:
        result['type'] = 'string'
    elif isinstance(enum_type, int) or types == {int}:
        result['type'] = 'integer'
    elif isinstance(enum_type, float) or types == {float}:
        result['type'] = 'numeric'
    elif types == {bool}:
        result['type'] = 'boolean'
    elif types == {list}:
        result['type'] = 'array'
    return result