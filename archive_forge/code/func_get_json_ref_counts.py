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
def get_json_ref_counts(self, json_schema: JsonSchemaValue) -> dict[JsonRef, int]:
    """Get all values corresponding to the key '$ref' anywhere in the json_schema."""
    json_refs: dict[JsonRef, int] = Counter()

    def _add_json_refs(schema: Any) -> None:
        if isinstance(schema, dict):
            if '$ref' in schema:
                json_ref = JsonRef(schema['$ref'])
                if not isinstance(json_ref, str):
                    return
                already_visited = json_ref in json_refs
                json_refs[json_ref] += 1
                if already_visited:
                    return
                defs_ref = self.json_to_defs_refs[json_ref]
                if defs_ref in self._core_defs_invalid_for_json_schema:
                    raise self._core_defs_invalid_for_json_schema[defs_ref]
                _add_json_refs(self.definitions[defs_ref])
            for v in schema.values():
                _add_json_refs(v)
        elif isinstance(schema, list):
            for v in schema:
                _add_json_refs(v)
    _add_json_refs(json_schema)
    return json_refs