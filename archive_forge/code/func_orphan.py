from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def orphan(self) -> None:
    """Detach this node from its parent."""
    self._set_parent(new_parent=None)