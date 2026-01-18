from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
@property
def siblings(self: Tree) -> dict[str, Tree]:
    """
        Nodes with the same parent as this node.
        """
    if self.parent:
        return {name: child for name, child in self.parent.children.items() if child is not self}
    else:
        return {}