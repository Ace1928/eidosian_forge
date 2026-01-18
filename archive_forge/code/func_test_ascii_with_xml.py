from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_ascii_with_xml(self):
    self.assertEqual(b'&amp;&apos;&quot;&lt;&gt;', breezy.bzr.xml_serializer.encode_and_escape('&\'"<>'))