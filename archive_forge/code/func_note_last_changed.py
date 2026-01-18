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
def note_last_changed(self, file_id, revision_id):
    if file_id in self._last_changed and self._last_changed[file_id] != revision_id:
        raise BzrError('Mismatched last-changed revision for file_id {%s}: %s != %s' % (file_id, self._last_changed[file_id], revision_id))
    self._last_changed[file_id] = revision_id