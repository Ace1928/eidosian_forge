from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from ..core.has_props import HasProps, Qualified
from ..util.dataclasses import entries, is_dataclass
def queue_one(obj: Model) -> None:
    if obj.id not in ids and (not (callable(discard) and discard(obj))):
        queued.append(obj)