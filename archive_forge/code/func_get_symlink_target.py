from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def get_symlink_target(self, path):
    try:
        ie = self._new_info_by_path[path]
    except KeyError:
        file_id = self.path2id(path)
        return self._basis_inv.get_entry(file_id).symlink_target
    else:
        return ie.symlink_target