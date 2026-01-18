import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_changes(self, creator, message=None):
    """Store the changes in a ShelfCreator on a shelf."""
    next_shelf, shelf_file = self.new_shelf()
    try:
        creator.write_shelf(shelf_file, message)
    finally:
        shelf_file.close()
    creator.transform()
    return next_shelf