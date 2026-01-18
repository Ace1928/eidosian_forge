from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def is_unmodified(self, other):
    other_revision = getattr(other, 'revision', None)
    if other_revision is None:
        return False
    return self.revision == other.revision