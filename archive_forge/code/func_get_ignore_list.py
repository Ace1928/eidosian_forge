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
def get_ignore_list(self):
    """Return list of ignore patterns.

        Cached in the Tree object after the first call.
        """
    ignoreset = getattr(self, '_ignoreset', None)
    if ignoreset is not None:
        return ignoreset
    ignore_globs = set()
    ignore_globs.update(ignores.get_runtime_ignores())
    ignore_globs.update(ignores.get_user_ignores())
    if self.has_filename(self._format.ignore_filename):
        with self.get_file(self._format.ignore_filename) as f:
            ignore_globs.update(ignores.parse_ignore_file(f))
    self._ignoreset = ignore_globs
    return ignore_globs