from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def get_idpath(self, file_id):
    """Return a list of file_ids for the path to an entry.

        The list contains one element for each directory followed by
        the id of the file itself.  So the length of the returned list
        is equal to the depth of the file in the tree, counting the
        root directory as depth 1.
        """
    p = []
    for parent in self._iter_file_id_parents(file_id):
        p.insert(0, parent.file_id)
    return p