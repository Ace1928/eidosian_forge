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
def get_flattened_anyof(self, schemas: list[JsonSchemaValue]) -> JsonSchemaValue:
    members = []
    for schema in schemas:
        if len(schema) == 1 and 'anyOf' in schema:
            members.extend(schema['anyOf'])
        else:
            members.append(schema)
    members = _deduplicate_schemas(members)
    if len(members) == 1:
        return members[0]
    return {'anyOf': members}