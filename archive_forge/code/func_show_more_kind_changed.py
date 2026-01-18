from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def show_more_kind_changed(item):
    to_file.write(' ({} => {})'.format(item.kind[0], item.kind[1]))