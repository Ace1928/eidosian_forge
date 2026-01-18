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
def get_ghosts(self):
    """Return a list of the parent tree revision ids that are ghosts."""
    self._read_header_if_needed()
    return self._ghosts