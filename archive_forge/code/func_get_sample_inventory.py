from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def get_sample_inventory(self):
    inv = Inventory(b'tree-root-321', revision_id=b'rev_outer')
    inv.add(inventory.InventoryFile(b'file-id', 'file', b'tree-root-321'))
    inv.add(inventory.InventoryDirectory(b'dir-id', 'dir', b'tree-root-321'))
    inv.add(inventory.InventoryLink(b'link-id', 'link', b'tree-root-321'))
    inv.get_entry(b'tree-root-321').revision = b'rev_outer'
    inv.get_entry(b'dir-id').revision = b'rev_outer'
    inv.get_entry(b'file-id').revision = b'rev_outer'
    inv.get_entry(b'file-id').text_sha1 = b'A'
    inv.get_entry(b'file-id').text_size = 1
    inv.get_entry(b'link-id').revision = b'rev_outer'
    inv.get_entry(b'link-id').symlink_target = 'a'
    return inv