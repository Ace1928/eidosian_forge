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
def copy_one(self, from_rel, to_rel):
    """Copy a file in the tree to a new location.

        This default implementation just copies the file, then
        adds the target.

        Args:
          from_rel: From location (relative to tree root)
          to_rel: Target location (relative to tree root)
        """
    import shutil
    shutil.copyfile(self.abspath(from_rel), self.abspath(to_rel))
    self.add(to_rel)