from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
def has_top_DIE(self):
    """ Returns whether the top DIE in this CU has already been parsed and cached.
            No parsing on demand!
        """
    return len(self._diemap) > 0