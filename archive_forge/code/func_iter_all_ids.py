from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def iter_all_ids(self):
    """Iterate over all file-ids."""
    for key, _ in self.id_to_entry.iteritems():
        yield key[-1]