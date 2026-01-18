from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
@classmethod
def from_inventory(klass, chk_store, inventory, maximum_size=0, search_key_name=b'plain'):
    """Create a CHKInventory from an existing inventory.

        The content of inventory is copied into the chk_store, and a
        CHKInventory referencing that is returned.

        :param chk_store: A CHK capable VersionedFiles instance.
        :param inventory: The inventory to copy.
        :param maximum_size: The CHKMap node size limit.
        :param search_key_name: The identifier for the search key function
        """
    result = klass(search_key_name)
    result.revision_id = inventory.revision_id
    result.root_id = inventory.root.file_id
    entry_to_bytes = result._entry_to_bytes
    parent_id_basename_key = result._parent_id_basename_key
    id_to_entry_dict = {}
    parent_id_basename_dict = {}
    for path, entry in inventory.iter_entries():
        key = StaticTuple(entry.file_id).intern()
        id_to_entry_dict[key] = entry_to_bytes(entry)
        p_id_key = parent_id_basename_key(entry)
        parent_id_basename_dict[p_id_key] = entry.file_id
    result._populate_from_dicts(chk_store, id_to_entry_dict, parent_id_basename_dict, maximum_size=maximum_size)
    return result