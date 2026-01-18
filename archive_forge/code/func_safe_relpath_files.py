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
def safe_relpath_files(self, file_list, canonicalize=True, apply_view=True):
    """Convert file_list into a list of relpaths in tree.

        Args:
          self: A tree to operate on.
          file_list: A list of user provided paths or None.
          apply_view: if True and a view is set, apply it or check that
            specified files are within it

        Returns:
          A list of relative paths.

        Raises:
          errors.PathNotChild: When a provided path is in a different self than
             self
        """
    if file_list is None:
        return None
    if self.supports_views() and apply_view:
        view_files = self.views.lookup_view()
    else:
        view_files = []
    new_list = []
    if canonicalize:

        def fixer(p):
            return osutils.canonical_relpath(self.basedir, p)
    else:
        fixer = self.relpath
    for filename in file_list:
        relpath = fixer(osutils.dereference_path(filename))
        if view_files and (not osutils.is_inside_any(view_files, relpath)):
            raise views.FileOutsideView(filename, view_files)
        new_list.append(relpath)
    return new_list