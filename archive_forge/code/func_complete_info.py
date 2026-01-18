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
def complete_info(self):
    """This makes sure that all information is properly
        split up, based on the assumptions that can be made
        when information is missing.
        """
    from breezy.timestamp import unpack_highres_date
    if not self.timestamp and self.date:
        self.timestamp, self.timezone = unpack_highres_date(self.date)
    self.real_revisions = []
    for rev in self.revisions:
        if rev.timestamp is None:
            if rev.date is not None:
                rev.timestamp, rev.timezone = unpack_highres_date(rev.date)
            else:
                rev.timestamp = self.timestamp
                rev.timezone = self.timezone
        if rev.message is None and self.message:
            rev.message = self.message
        if rev.committer is None and self.committer:
            rev.committer = self.committer
        self.real_revisions.append(rev.as_revision())