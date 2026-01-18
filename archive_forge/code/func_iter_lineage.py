from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def iter_lineage(self: Tree) -> tuple[Tree, ...]:
    """Iterate up the tree, starting from the current node."""
    from warnings import warn
    warn('`iter_lineage` has been deprecated, and in the future will raise an error.Please use `parents` from now on.', DeprecationWarning)
    return tuple((self, *self.parents))