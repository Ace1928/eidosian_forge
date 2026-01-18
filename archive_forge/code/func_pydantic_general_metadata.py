from __future__ import annotations as _annotations
import dataclasses
import sys
import warnings
from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from pydantic_core import PydanticUndefined
from pydantic.errors import PydanticUserError
from . import _typing_extra
from ._config import ConfigWrapper
from ._repr import Representation
from ._typing_extra import get_cls_type_hints_lenient, get_type_hints, is_classvar, is_finalvar
def pydantic_general_metadata(**metadata: Any) -> BaseMetadata:
    """Create a new `_PydanticGeneralMetadata` class with the given metadata.

    Args:
        **metadata: The metadata to add.

    Returns:
        The new `_PydanticGeneralMetadata` class.
    """
    return _general_metadata_cls()(metadata)