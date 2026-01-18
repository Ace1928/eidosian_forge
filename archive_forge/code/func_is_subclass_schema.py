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
def is_subclass_schema(self, schema: core_schema.IsSubclassSchema) -> JsonSchemaValue:
    """Handles JSON schema generation for a core schema that checks if a value is a subclass of a class.

        For backwards compatibility with v1, this does not raise an error, but can be overridden to change this.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    return {}