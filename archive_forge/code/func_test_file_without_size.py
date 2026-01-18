from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_file_without_size(self):
    file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
    file_entry.text_sha1 = b'foo'
    self.assertRaises(InventoryDeltaError, inventory_delta._file_content, file_entry)