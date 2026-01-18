from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_empty_delta_to_lines(self):
    old_inv = Inventory(None)
    new_inv = Inventory(None)
    delta = new_inv._make_delta(old_inv)
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
    self.assertEqual(BytesIO(empty_lines).readlines(), serializer.delta_to_lines(NULL_REVISION, NULL_REVISION, delta))