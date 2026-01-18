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
def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that defines a model.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    cls = cast('type[BaseModel]', schema['cls'])
    config = cls.model_config
    title = config.get('title')
    with self._config_wrapper_stack.push(config):
        json_schema = self.generate_inner(schema['schema'])
    json_schema_extra = config.get('json_schema_extra')
    if cls.__pydantic_root_model__:
        root_json_schema_extra = cls.model_fields['root'].json_schema_extra
        if json_schema_extra and root_json_schema_extra:
            raise ValueError('"model_config[\'json_schema_extra\']" and "Field.json_schema_extra" on "RootModel.root" field must not be set simultaneously')
        if root_json_schema_extra:
            json_schema_extra = root_json_schema_extra
    json_schema = self._update_class_schema(json_schema, title, config.get('extra', None), cls, json_schema_extra)
    return json_schema