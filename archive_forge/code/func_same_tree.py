from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def same_tree(self, other: Tree) -> bool:
    """True if other node is in the same tree as this node."""
    return self.root is other.root