from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
@property
def lineage(self: Tree) -> tuple[Tree, ...]:
    """All parent nodes and their parent nodes, starting with the closest."""
    from warnings import warn
    warn('`lineage` has been deprecated, and in the future will raise an error.Please use `parents` from now on.', DeprecationWarning)
    return self.iter_lineage()