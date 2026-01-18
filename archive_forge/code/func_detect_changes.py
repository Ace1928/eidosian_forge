from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def detect_changes(self, old_entry):
    """See InventoryEntry.detect_changes."""
    text_modified = self.symlink_target != old_entry.symlink_target
    if text_modified:
        trace.mutter('    symlink target changed')
    meta_modified = False
    return (text_modified, meta_modified)