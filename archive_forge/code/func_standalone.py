from __future__ import annotations
import abc
import enum
import typing
from .. import types
@property
def standalone(self) -> bool:
    """Whether this :class:`Var` is a standalone variable that owns its storage location.
        This is currently always ``False``, but will expand in the future to support memory-owning
        storage locations."""
    return False