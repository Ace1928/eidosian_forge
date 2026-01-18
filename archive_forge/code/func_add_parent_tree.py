import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple
import breezy
from .lazy_import import lazy_import
import stat
from breezy import (
from . import errors, mutabletree, osutils
from . import revision as _mod_revision
from .controldir import (ControlComponent, ControlComponentFormat,
from .i18n import gettext
from .symbol_versioning import deprecated_in, deprecated_method
from .trace import mutter, note
from .transport import NoSuchFile
def add_parent_tree(self, parent_tuple, allow_leftmost_as_ghost=False):
    """Add revision_id, tree tuple as a parent.

        This is equivalent to retrieving the current list of parent trees
        and setting the list to its value plus parent_tuple. See also
        add_parent_tree_id - if you only have a parent id available it will be
        simpler to use that api. If you have the parent already available, using
        this api is preferred.

        Args:
          parent_tuple: The (revision id, tree) to add to the parent list.
            If the revision_id is a ghost, pass None for the tree.
          allow_leftmost_as_ghost: Allow the first parent to be a ghost.
        """
    with self.lock_tree_write():
        parent_ids = self.get_parent_ids() + [parent_tuple[0]]
        if len(parent_ids) > 1:
            allow_leftmost_as_ghost = True
        self.set_parent_ids(parent_ids, allow_leftmost_as_ghost=allow_leftmost_as_ghost)