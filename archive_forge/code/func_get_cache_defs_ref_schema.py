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
def get_cache_defs_ref_schema(self, core_ref: CoreRef) -> tuple[DefsRef, JsonSchemaValue]:
    """This method wraps the get_defs_ref method with some cache-lookup/population logic,
        and returns both the produced defs_ref and the JSON schema that will refer to the right definition.

        Args:
            core_ref: The core reference to get the definitions reference for.

        Returns:
            A tuple of the definitions reference and the JSON schema that will refer to it.
        """
    core_mode_ref = (core_ref, self.mode)
    maybe_defs_ref = self.core_to_defs_refs.get(core_mode_ref)
    if maybe_defs_ref is not None:
        json_ref = self.core_to_json_refs[core_mode_ref]
        return (maybe_defs_ref, {'$ref': json_ref})
    defs_ref = self.get_defs_ref(core_mode_ref)
    self.core_to_defs_refs[core_mode_ref] = defs_ref
    self.defs_to_core_refs[defs_ref] = core_mode_ref
    json_ref = JsonRef(self.ref_template.format(model=defs_ref))
    self.core_to_json_refs[core_mode_ref] = json_ref
    self.json_to_defs_refs[json_ref] = defs_ref
    ref_json_schema = {'$ref': json_ref}
    return (defs_ref, ref_json_schema)