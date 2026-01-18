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
def set_root_id(self, file_id):
    """Set the root id for this tree."""
    if not self.supports_setting_file_ids():
        raise SettingFileIdUnsupported()
    with self.lock_tree_write():
        if file_id is None:
            raise ValueError('WorkingTree.set_root_id with fileid=None')
        self._set_root_id(file_id)