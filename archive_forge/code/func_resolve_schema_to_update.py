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
def resolve_schema_to_update(self, json_schema: JsonSchemaValue) -> JsonSchemaValue:
    """Resolve a JsonSchemaValue to the non-ref schema if it is a $ref schema.

        Args:
            json_schema: The schema to resolve.

        Returns:
            The resolved schema.
        """
    if '$ref' in json_schema:
        schema_to_update = self.get_schema_from_definitions(JsonRef(json_schema['$ref']))
        if schema_to_update is None:
            raise RuntimeError(f'Cannot update undefined schema for $ref={json_schema['$ref']}')
        return self.resolve_schema_to_update(schema_to_update)
    else:
        schema_to_update = json_schema
    return schema_to_update