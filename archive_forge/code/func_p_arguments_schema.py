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
def p_arguments_schema(self, arguments: list[core_schema.ArgumentsParameter], var_args_schema: CoreSchema | None) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a function's positional arguments.

        Args:
            arguments: The core schema.

        Returns:
            The generated JSON schema.
        """
    prefix_items: list[JsonSchemaValue] = []
    min_items = 0
    for argument in arguments:
        name = self.get_argument_name(argument)
        argument_schema = self.generate_inner(argument['schema']).copy()
        argument_schema['title'] = self.get_title_from_name(name)
        prefix_items.append(argument_schema)
        if argument['schema']['type'] != 'default':
            min_items += 1
    json_schema: JsonSchemaValue = {'type': 'array', 'prefixItems': prefix_items}
    if min_items:
        json_schema['minItems'] = min_items
    if var_args_schema:
        items_schema = self.generate_inner(var_args_schema)
        if items_schema:
            json_schema['items'] = items_schema
    else:
        json_schema['maxItems'] = len(prefix_items)
    return json_schema