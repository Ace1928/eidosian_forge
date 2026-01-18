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
def extra_info(info, new_path):
    last_changed = None
    encoding = None
    for info_item in info:
        try:
            name, value = info_item.split(':', 1)
        except ValueError:
            raise ValueError('Value %r has no colon' % info_item)
        if name == 'last-changed':
            last_changed = value
        elif name == 'executable':
            val = value == 'yes'
            bundle_tree.note_executable(new_path, val)
        elif name == 'target':
            bundle_tree.note_target(new_path, value)
        elif name == 'encoding':
            encoding = value
    return (last_changed, encoding)