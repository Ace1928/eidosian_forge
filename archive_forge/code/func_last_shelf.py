import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def last_shelf(self):
    """Return the id of the last-created shelved change."""
    active = self.active_shelves()
    if len(active) > 0:
        return active[-1]
    else:
        return None