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
def auto_resolve(self):
    """Automatically resolve text conflicts according to contents.

        Only text conflicts are auto_resolvable. Files with no conflict markers
        are considered 'resolved', because bzr always puts conflict markers
        into files that have text conflicts.  The corresponding .THIS .BASE and
        .OTHER files are deleted, as per 'resolve'.

        Returns: a tuple of lists: (un_resolved, resolved).
        """
    with self.lock_tree_write():
        un_resolved = []
        resolved = []
        for conflict in self.conflicts():
            try:
                conflict.action_auto(self)
            except NotImplementedError:
                un_resolved.append(conflict)
            else:
                conflict.cleanup(self)
                resolved.append(conflict)
        self.set_conflicts(un_resolved)
        return (un_resolved, resolved)