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
def note_rename(self, old_path, new_path):
    """A file/directory has been renamed from old_path => new_path"""
    if new_path in self._renamed:
        raise AssertionError(new_path)
    if old_path in self._renamed_r:
        raise AssertionError(old_path)
    self._renamed[new_path] = old_path
    self._renamed_r[old_path] = new_path