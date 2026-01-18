import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def scan_down(self, top, left, right):
    """
        Look for the bottom-right corner of the cell, making note of all row
        boundaries.
        """
    rowseps = {}
    for i in range(top + 1, self.bottom + 1):
        if self.block[i][right] == '+':
            rowseps[i] = [right]
            result = self.scan_left(top, left, i, right)
            if result:
                newrowseps, colseps = result
                update_dict_of_lists(rowseps, newrowseps)
                return (i, rowseps, colseps)
        elif self.block[i][right] != '|':
            return None
    return None