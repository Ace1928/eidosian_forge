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