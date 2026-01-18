import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
def test_canned_inventory(self):
    """Test unpacked a canned inventory v4 file."""
    inp = BytesIO(_working_inventory_v4)
    inv = xml4.serializer_v4.read_inventory(inp)
    self.assertEqual(len(inv), 4)
    self.assertTrue(inv.has_id(b'bar-20050901064931-73b4b1138abc9cd2'))