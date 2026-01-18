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
def field_is_present(self, field: CoreSchemaField) -> bool:
    """Whether the field should be included in the generated JSON schema.

        Args:
            field: The schema for the field itself.

        Returns:
            `True` if the field should be included in the generated JSON schema, `False` otherwise.
        """
    if self.mode == 'serialization':
        return not field.get('serialization_exclude')
    elif self.mode == 'validation':
        return True
    else:
        assert_never(self.mode)