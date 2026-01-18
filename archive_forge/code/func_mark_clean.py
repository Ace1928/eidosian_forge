import os
from typing import Set
from pyinotify import (IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE,
from .workingtree import WorkingTree
def mark_clean(self) -> None:
    """Mark the subtree as not having any changes."""
    self._process_pending()
    self._process.paths.clear()
    self._process.created.clear()