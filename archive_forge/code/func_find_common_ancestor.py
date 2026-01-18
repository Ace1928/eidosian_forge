from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def find_common_ancestor(self, other: NamedNode) -> NamedNode:
    """
        Find the first common ancestor of two nodes in the same tree.

        Raise ValueError if they are not in the same tree.
        """
    if self is other:
        return self
    other_paths = [op.path for op in other.parents]
    for parent in (self, *self.parents):
        if parent.path in other_paths:
            return parent
    raise NotFoundInTreeError('Cannot find common ancestor because nodes do not lie within the same tree')