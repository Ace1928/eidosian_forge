import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def is_snapshot(self):
    """Return true of this hunk is effectively a fulltext"""
    if len(self.hunks) != 1:
        return False
    return isinstance(self.hunks[0], NewText)