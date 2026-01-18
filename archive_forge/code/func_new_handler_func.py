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
def new_handler_func(schema_or_field: CoreSchemaOrField, current_handler: GetJsonSchemaHandler=current_handler, js_modify_function: GetJsonSchemaFunction=js_modify_function) -> JsonSchemaValue:
    json_schema = js_modify_function(schema_or_field, current_handler)
    if _core_utils.is_core_schema(schema_or_field):
        json_schema = populate_defs(schema_or_field, json_schema)
        json_schema = convert_to_all_of(json_schema)
    return json_schema