from __future__ import annotations as _annotations
import operator
import sys
import types
import typing
import warnings
from copy import copy, deepcopy
from typing import Any, ClassVar
import pydantic_core
import typing_extensions
from pydantic_core import PydanticUndefined
from ._internal import (
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation, PydanticUserError
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue, model_json_schema
from .warnings import PydanticDeprecatedSince20
@classmethod
def model_validate_strings(cls: type[Model], obj: Any, *, strict: bool | None=None, context: dict[str, Any] | None=None) -> Model:
    """Validate the given object contains string data against the Pydantic model.

        Args:
            obj: The object contains string data to validate.
            strict: Whether to enforce types strictly.
            context: Extra variables to pass to the validator.

        Returns:
            The validated Pydantic model.
        """
    __tracebackhide__ = True
    return cls.__pydantic_validator__.validate_strings(obj, strict=strict, context=context)