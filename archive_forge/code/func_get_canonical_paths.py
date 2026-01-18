import errno
import itertools
import operator
import os
import stat
import sys
from bisect import bisect_left
from collections import deque
from io import BytesIO
import breezy
from .. import lazy_import
from . import bzrdir
import contextlib
from breezy import (
from breezy.bzr import (
from .. import errors, osutils
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..lock import LogicalLockResult
from ..trace import mutter, note
from ..tree import (MissingNestedTree, TreeDirectory, TreeEntry, TreeFile,
from ..workingtree import WorkingTree, WorkingTreeFormat, format_registry
from .inventorytree import InventoryRevisionTree, MutableInventoryTree
def get_canonical_paths(self, paths):
    """Look up canonical paths for multiple items.

        :param paths: A sequence of paths relative to the root of the tree.
        :return: A iterator over paths, with each item the corresponding input
            path adjusted to account for existing elements that match case
            insensitively.
        """
    with self.lock_read():
        if not self.case_sensitive:

            def normalize(x):
                return x.lower()
        elif sys.platform == 'darwin':
            import unicodedata

            def normalize(x):
                return unicodedata.normalize('NFC', x)
        else:
            normalize = None
        for path in paths:
            if normalize is None or self.is_versioned(path):
                yield path.strip('/')
            else:
                yield get_canonical_path(self, path, normalize)