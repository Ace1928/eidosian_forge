from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_inventory_text_v8(self):
    inv = self.get_sample_inventory()
    lines = xml8.serializer_v8.write_inventory_to_lines(inv)
    self.assertEqualDiff(_expected_inv_v8, b''.join(lines))