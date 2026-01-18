from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def has_changed(self):
    return bool(self.modified or self.added or self.removed or self.renamed or self.copied or self.kind_changed)