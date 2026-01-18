from __future__ import annotations
import logging # isort:skip
import contextlib
import weakref
from typing import (
from ..core.types import ID
from ..model import Model
from ..util.datatypes import MultiValuedDict
@property
def synced_references(self) -> set[Model]:
    return set((model for model in self._models.values() if model not in self._new_models))