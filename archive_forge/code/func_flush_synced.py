from __future__ import annotations
import logging # isort:skip
import contextlib
import weakref
from typing import (
from ..core.types import ID
from ..model import Model
from ..util.datatypes import MultiValuedDict
def flush_synced(self, is_still_new: Callable[[Model], bool] | None=None) -> None:
    """ Clean up transient state of the document's models. """
    if is_still_new is None:
        self._new_models.clear()
    else:
        self._new_models = set((new_model for new_model in self._new_models if is_still_new(new_model)))