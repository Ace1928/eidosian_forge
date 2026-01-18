from io import BytesIO
import fastbencode as bencode
from .. import lazy_import
from breezy.bzr import (
from .. import cache_utf8, errors
from .. import revision as _mod_revision
from . import serializer
def read_inventory(self, f, revision_id=None):
    """Read an inventory from a file-like object."""
    try:
        try:
            return self._unpack_inventory(self._read_element(f), revision_id=None)
        finally:
            f.close()
    except xml_serializer.ParseError as e:
        raise serializer.UnexpectedInventoryFormat(e)