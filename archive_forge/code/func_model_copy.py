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
def model_copy(self: Model, *, update: dict[str, Any] | None=None, deep: bool=False) -> Model:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/serialization/#model_copy

        Returns a copy of the model.

        Args:
            update: Values to change/add in the new model. Note: the data is not validated
                before creating the new model. You should trust this data.
            deep: Set to `True` to make a deep copy of the model.

        Returns:
            New model instance.
        """
    copied = self.__deepcopy__() if deep else self.__copy__()
    if update:
        if self.model_config.get('extra') == 'allow':
            for k, v in update.items():
                if k in self.model_fields:
                    copied.__dict__[k] = v
                else:
                    if copied.__pydantic_extra__ is None:
                        copied.__pydantic_extra__ = {}
                    copied.__pydantic_extra__[k] = v
        else:
            copied.__dict__.update(update)
        copied.__pydantic_fields_set__.update(update.keys())
    return copied