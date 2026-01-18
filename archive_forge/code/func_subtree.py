from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
@property
def subtree(self: Tree) -> Iterator[Tree]:
    """
        An iterator over all nodes in this tree, including both self and all descendants.

        Iterates depth-first.

        See Also
        --------
        DataTree.descendants
        """
    from xarray.datatree_.datatree import iterators
    return iterators.PreOrderIter(self)