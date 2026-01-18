import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def set_sibling_indices(self, sibling_combined_graph_indices):
    """Set the CombinedGraphIndex objects to reorder after reordering self.
        """
    self._sibling_indices = sibling_combined_graph_indices