from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
def iter_DIEs(self):
    """ Iterate over all the DIEs in the CU, in order of their appearance.
            Note that null DIEs will also be returned.
        """
    return self._iter_DIE_subtree(self.get_top_DIE())