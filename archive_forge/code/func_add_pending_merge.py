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
def add_pending_merge(self, *revision_ids):
    with self.lock_tree_write():
        parents = self.get_parent_ids()
        updated = False
        for rev_id in revision_ids:
            if rev_id in parents:
                continue
            parents.append(rev_id)
            updated = True
        if updated:
            self.set_parent_ids(parents, allow_leftmost_as_ghost=True)