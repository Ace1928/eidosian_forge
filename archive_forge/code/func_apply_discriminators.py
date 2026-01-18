from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def apply_discriminators(schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
    definitions: dict[str, CoreSchema] | None = None

    def inner(s: core_schema.CoreSchema, recurse: _core_utils.Recurse) -> core_schema.CoreSchema:
        nonlocal definitions
        s = recurse(s, inner)
        if s['type'] == 'tagged-union':
            return s
        metadata = s.get('metadata', {})
        discriminator = metadata.pop(CORE_SCHEMA_METADATA_DISCRIMINATOR_PLACEHOLDER_KEY, None)
        if discriminator is not None:
            if definitions is None:
                definitions = collect_definitions(schema)
            s = apply_discriminator(s, discriminator, definitions)
        return s
    return simplify_schema_references(_core_utils.walk_core_schema(schema, inner))