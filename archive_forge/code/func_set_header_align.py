from __future__ import division
import sys
import unicodedata
from functools import reduce
def set_header_align(self, array):
    """Set the desired header alignment

        - the elements of the array should be either "l", "c" or "r":

            * "l": column flushed left
            * "c": column centered
            * "r": column flushed right
        """
    self._check_row_size(array)
    self._header_align = array
    return self