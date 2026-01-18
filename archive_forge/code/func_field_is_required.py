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
def field_is_required(self, field: core_schema.ModelField | core_schema.DataclassField | core_schema.TypedDictField, total: bool) -> bool:
    """Whether the field should be marked as required in the generated JSON schema.
        (Note that this is irrelevant if the field is not present in the JSON schema.).

        Args:
            field: The schema for the field itself.
            total: Only applies to `TypedDictField`s.
                Indicates if the `TypedDict` this field belongs to is total, in which case any fields that don't
                explicitly specify `required=False` are required.

        Returns:
            `True` if the field should be marked as required in the generated JSON schema, `False` otherwise.
        """
    if self.mode == 'serialization' and self._config.json_schema_serialization_defaults_required:
        return not field.get('serialization_exclude')
    elif field['type'] == 'typed-dict-field':
        return field.get('required', total)
    else:
        return field['schema']['type'] != 'default'