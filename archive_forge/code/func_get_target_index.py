from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
def get_target_index(name: str, target_indexes: TargetIndexes) -> int:
    """Find or add the target in the result set and return the target index."""
    return target_indexes.setdefault(name, len(target_indexes))