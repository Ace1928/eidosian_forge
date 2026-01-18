import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def insert_index(self, pos, index, name=None):
    """Insert a new index in the list of indices to query.

        :param pos: The position to insert the index.
        :param index: The index to insert.
        :param name: a name for this index, e.g. a pack name.  These names can
            be used to reflect index reorderings to related CombinedGraphIndex
            instances that use the same names.  (see set_sibling_indices)
        """
    self._indices.insert(pos, index)
    self._index_names.insert(pos, name)