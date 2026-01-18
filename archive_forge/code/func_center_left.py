from __future__ import annotations
import logging # isort:skip
from typing import Any, ClassVar, Literal
from ..core.properties import (
from ..core.property.aliases import CoordinateLike
from ..model import Model
@property
def center_left(self) -> Node:
    return self._node('center_left')