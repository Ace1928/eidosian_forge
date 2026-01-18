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
def get_rev_id(last_changed, path, kind):
    if last_changed is not None:
        changed_revision_id = cache_utf8.encode(last_changed)
    else:
        changed_revision_id = revision_id
    bundle_tree.note_last_changed(path, changed_revision_id)
    return changed_revision_id