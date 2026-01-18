from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def make_entry(kind, name, parent_id, file_id, **attrs):
    entry = inventory.make_entry(kind, name, parent_id, file_id)
    for name, value in attrs.items():
        setattr(entry, name, value)
    return entry