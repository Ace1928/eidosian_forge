from __future__ import annotations
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from .._utils import LazyProxy
from ._common import MissingDependencyError, format_instructions
def has_numpy() -> bool:
    try:
        import numpy
    except ImportError:
        return False
    return True