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
def get_size_and_sha1(self, new_path):
    """Return the size and sha1 hash of the given file id.
        If the file was not locally modified, this is extracted
        from the base_tree. Rather than re-reading the file.
        """
    if new_path is None:
        return (None, None)
    if new_path not in self.patches:
        base_path = self.old_path(new_path)
        text_size = self.base_tree.get_file_size(base_path)
        text_sha1 = self.base_tree.get_file_sha1(base_path)
        return (text_size, text_sha1)
    fileobj = self.get_file(new_path)
    content = fileobj.read()
    return (len(content), sha_string(content))