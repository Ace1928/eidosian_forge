from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def id2path(self, file_id, recurse='down'):
    if file_id in self._new_info_by_id:
        new_path = self._new_info_by_id[file_id][0]
        if new_path is None:
            raise errors.NoSuchId(self, file_id)
        return new_path
    return self._basis_inv.id2path(file_id)