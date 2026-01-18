import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
def by_parent(self):
    """Return a map of parent: children for known parents.

        Only new paths and parents of tree files with assigned ids are used.
        """
    by_parent = {}
    items = list(self._new_parent.items())
    items.extend(((t, self.final_parent(t)) for t in list(self._tree_id_paths)))
    for trans_id, parent_id in items:
        if parent_id not in by_parent:
            by_parent[parent_id] = set()
        by_parent[parent_id].add(trans_id)
    return by_parent