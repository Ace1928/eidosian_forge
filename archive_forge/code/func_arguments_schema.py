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
def arguments_schema(self, schema: core_schema.ArgumentsSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a function's arguments.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    metadata = _core_metadata.CoreMetadataHandler(schema).metadata
    prefer_positional = metadata.get('pydantic_js_prefer_positional_arguments')
    arguments = schema['arguments_schema']
    kw_only_arguments = [a for a in arguments if a.get('mode') == 'keyword_only']
    kw_or_p_arguments = [a for a in arguments if a.get('mode') in {'positional_or_keyword', None}]
    p_only_arguments = [a for a in arguments if a.get('mode') == 'positional_only']
    var_args_schema = schema.get('var_args_schema')
    var_kwargs_schema = schema.get('var_kwargs_schema')
    if prefer_positional:
        positional_possible = not kw_only_arguments and (not var_kwargs_schema)
        if positional_possible:
            return self.p_arguments_schema(p_only_arguments + kw_or_p_arguments, var_args_schema)
    keyword_possible = not p_only_arguments and (not var_args_schema)
    if keyword_possible:
        return self.kw_arguments_schema(kw_or_p_arguments + kw_only_arguments, var_kwargs_schema)
    if not prefer_positional:
        positional_possible = not kw_only_arguments and (not var_kwargs_schema)
        if positional_possible:
            return self.p_arguments_schema(p_only_arguments + kw_or_p_arguments, var_args_schema)
    raise PydanticInvalidForJsonSchema('Unable to generate JSON schema for arguments validator with positional-only and keyword-only arguments')