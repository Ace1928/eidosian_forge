from __future__ import annotations as _annotations
from typing import TYPE_CHECKING, Any, Hashable, Sequence
from pydantic_core import CoreSchema, core_schema
from ..errors import PydanticUserError
from . import _core_utils
from ._core_utils import (
def set_discriminator_in_metadata(schema: CoreSchema, discriminator: Any) -> None:
    schema.setdefault('metadata', {})
    metadata = schema.get('metadata')
    assert metadata is not None
    metadata[CORE_SCHEMA_METADATA_DISCRIMINATOR_PLACEHOLDER_KEY] = discriminator