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
def kw_arguments_schema(self, arguments: list[core_schema.ArgumentsParameter], var_kwargs_schema: CoreSchema | None) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a function's keyword arguments.

        Args:
            arguments: The core schema.

        Returns:
            The generated JSON schema.
        """
    properties: dict[str, JsonSchemaValue] = {}
    required: list[str] = []
    for argument in arguments:
        name = self.get_argument_name(argument)
        argument_schema = self.generate_inner(argument['schema']).copy()
        argument_schema['title'] = self.get_title_from_name(name)
        properties[name] = argument_schema
        if argument['schema']['type'] != 'default':
            required.append(name)
    json_schema: JsonSchemaValue = {'type': 'object', 'properties': properties}
    if required:
        json_schema['required'] = required
    if var_kwargs_schema:
        additional_properties_schema = self.generate_inner(var_kwargs_schema)
        if additional_properties_schema:
            json_schema['additionalProperties'] = additional_properties_schema
    else:
        json_schema['additionalProperties'] = False
    return json_schema