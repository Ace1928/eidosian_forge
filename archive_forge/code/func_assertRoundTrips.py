from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def assertRoundTrips(self, xml_string):
    inp = BytesIO(xml_string)
    inv = breezy.bzr.xml5.serializer_v5.read_inventory(inp)
    outp = BytesIO()
    breezy.bzr.xml5.serializer_v5.write_inventory(inv, outp)
    self.assertEqualDiff(xml_string, outp.getvalue())
    lines = breezy.bzr.xml5.serializer_v5.write_inventory_to_lines(inv)
    outp.seek(0)
    self.assertEqual(outp.readlines(), lines)
    inv2 = breezy.bzr.xml5.serializer_v5.read_inventory(BytesIO(outp.getvalue()))
    self.assertEqual(inv, inv2)