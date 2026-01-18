from __future__ import division
import sys
import unicodedata
from functools import reduce
def set_cols_valign(self, array):
    """Set the desired columns vertical alignment

        - the elements of the array should be either "t", "m" or "b":

            * "t": column aligned on the top of the cell
            * "m": column aligned on the middle of the cell
            * "b": column aligned on the bottom of the cell
        """
    self._check_row_size(array)
    self._valign = array
    return self