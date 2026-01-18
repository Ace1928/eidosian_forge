from __future__ import annotations
import logging # isort:skip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ..core.types import PathLike
from ..document import Document
from ..resources import Resources, ResourcesMode
def curstate() -> State:
    """ Return the current State object

    Returns:
      State : the current default State object

    """
    global _STATE
    if _STATE is None:
        _STATE = State()
    return _STATE