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
def sorted_path_id(self):
    paths = []
    for result in self._new_id.items():
        paths.append(result)
    for id in self.base_tree.all_file_ids():
        try:
            path = self.id2path(id, recurse='none')
        except NoSuchId:
            continue
        paths.append((path, id))
    paths.sort()
    return paths