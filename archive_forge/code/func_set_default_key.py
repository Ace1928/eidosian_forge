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
def set_default_key(self, format_string):
    """Set the default format by its format string."""
    self._default_format_key = format_string
    self._default_format = None