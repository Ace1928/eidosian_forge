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
def json_or_python_schema(self, schema: core_schema.JsonOrPythonSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that allows values matching either the JSON schema or the
        Python schema.

        The JSON schema is used instead of the Python schema. If you want to use the Python schema, you should override
        this method.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    return self.generate_inner(schema['json_schema'])