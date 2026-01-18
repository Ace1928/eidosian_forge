from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_file_executable(self):
    file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
    file_entry.executable = True
    file_entry.text_sha1 = b'foo'
    file_entry.text_size = 10
    self.assertEqual(b'file\x0010\x00Y\x00foo', inventory_delta._file_content(file_entry))