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
def read_working_inventory(self):
    """Read the working inventory.

        :raises errors.InventoryModified: read_working_inventory will fail
            when the current in memory inventory has been modified.
        """
    with self.lock_read():
        if self._inventory_is_modified:
            raise InventoryModified(self)
        with self._transport.get('inventory') as f:
            result = self._deserialize(f)
        self._set_inventory(result, dirty=False)
        return result