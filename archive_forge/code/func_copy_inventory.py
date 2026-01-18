from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def copy_inventory(inv):
    entries = inv.iter_entries_by_dir()
    inv = inventory.Inventory(None, inv.revision_id)
    for path, inv_entry in entries:
        inv.add(inv_entry.copy())
    return inv