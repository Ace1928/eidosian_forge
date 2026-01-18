from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def get_file_with_stat(self, path):
    content = self.get_file_text(path)
    sio = BytesIO(content)
    return (sio, None)