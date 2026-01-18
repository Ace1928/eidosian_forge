import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
def old_path(self, new_path):
    """Get the old_path (path in the base_tree) for the file at new_path"""
    if new_path[:1] in ('\\', '/'):
        raise ValueError(new_path)
    old_path = self._renamed.get(new_path)
    if old_path is not None:
        return old_path
    dirname, basename = os.path.split(new_path)
    if dirname != '':
        old_dir = self.old_path(dirname)
        if old_dir is None:
            old_path = None
        else:
            old_path = pathjoin(old_dir, basename)
    else:
        old_path = new_path
    if old_path in self._renamed_r:
        return None
    return old_path