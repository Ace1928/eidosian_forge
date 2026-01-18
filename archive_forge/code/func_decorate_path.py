from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def decorate_path(path, kind, meta_modified=None):
    if not classify:
        return path
    if kind == 'directory':
        path += '/'
    elif kind == 'symlink':
        path += '@'
    if meta_modified:
        path += '*'
    return path