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
@staticmethod
def from_revision(revision):
    revision_info = RevisionInfo(revision.revision_id)
    date = timestamp.format_highres_date(revision.timestamp, revision.timezone)
    revision_info.date = date
    revision_info.timezone = revision.timezone
    revision_info.timestamp = revision.timestamp
    revision_info.message = revision.message.split('\n')
    revision_info.properties = [': '.join(p) for p in revision.properties.items()]
    return revision_info