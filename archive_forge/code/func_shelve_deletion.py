import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_deletion(self, file_id):
    """Shelve deletion of a file.

        This handles content and inventory id.
        :param file_id: The file_id of the file to shelve deletion of.
        """
    kind, name, parent, versioned = self.deletion[file_id]
    existing_path = self.target_tree.id2path(file_id)
    if not self.work_tree.has_filename(existing_path):
        existing_path = None
    version = not versioned[1]
    self._shelve_creation(self.target_tree, file_id, self.shelf_transform, self.work_transform, kind, name, parent, version, existing_path=existing_path)