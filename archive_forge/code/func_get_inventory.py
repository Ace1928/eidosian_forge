from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def get_inventory(self, revision_id):
    """Get a stored inventory."""
    return self.repo.get_inventory(revision_id)