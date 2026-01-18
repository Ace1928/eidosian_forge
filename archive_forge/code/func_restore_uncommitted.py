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
def restore_uncommitted(self):
    """Restore uncommitted changes from the branch into the tree."""
    with self.lock_write():
        unshelver = self.branch.get_unshelver(self)
        if unshelver is None:
            return
        try:
            merger = unshelver.make_merger()
            merger.ignore_zero = True
            merger.do_merge()
            self.branch.store_uncommitted(None)
        finally:
            unshelver.finalize()