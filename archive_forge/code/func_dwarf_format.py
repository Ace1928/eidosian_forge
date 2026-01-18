from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
def dwarf_format(self):
    """ Get the DWARF format (32 or 64) for this CU
        """
    return self.structs.dwarf_format