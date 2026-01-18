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
def literal_schema(self, schema: core_schema.LiteralSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a literal value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    expected = [v.value if isinstance(v, Enum) else v for v in schema['expected']]
    expected = [to_jsonable_python(v) for v in expected]
    if len(expected) == 1:
        return {'const': expected[0]}
    types = {type(e) for e in expected}
    if types == {str}:
        return {'enum': expected, 'type': 'string'}
    elif types == {int}:
        return {'enum': expected, 'type': 'integer'}
    elif types == {float}:
        return {'enum': expected, 'type': 'number'}
    elif types == {bool}:
        return {'enum': expected, 'type': 'boolean'}
    elif types == {list}:
        return {'enum': expected, 'type': 'array'}
    else:
        return {'enum': expected}