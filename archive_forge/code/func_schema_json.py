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
@typing_extensions.deprecated('The `schema_json` method is deprecated; use `model_json_schema` and json.dumps instead.', category=None)
def schema_json(cls, *, by_alias: bool=True, ref_template: str=DEFAULT_REF_TEMPLATE, **dumps_kwargs: Any) -> str:
    warnings.warn('The `schema_json` method is deprecated; use `model_json_schema` and json.dumps instead.', category=PydanticDeprecatedSince20)
    import json
    from .deprecated.json import pydantic_encoder
    return json.dumps(cls.model_json_schema(by_alias=by_alias, ref_template=ref_template), default=pydantic_encoder, **dumps_kwargs)