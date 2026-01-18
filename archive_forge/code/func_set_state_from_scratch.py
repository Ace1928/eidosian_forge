import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def set_state_from_scratch(self, working_inv, parent_trees, parent_ghosts):
    """Wipe the currently stored state and set it to something new.

        This is a hard-reset for the data we are working with.
        """
    self._requires_lock()
    empty_root = ((b'', b'', inventory.ROOT_ID), [(b'd', b'', 0, False, DirState.NULLSTAT)])
    empty_tree_dirblocks = [(b'', [empty_root]), (b'', [])]
    self._set_data([], empty_tree_dirblocks)
    self.set_state_from_inventory(working_inv)
    self.set_parent_trees(parent_trees, parent_ghosts)