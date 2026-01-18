import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def scan_left(self, top, left, bottom, right):
    """
        Noting column boundaries, look for the bottom-left corner of the cell.
        It must line up with the starting point.
        """
    colseps = {}
    line = self.block[bottom]
    for i in range(right - 1, left, -1):
        if line[i] == '+':
            colseps[i] = [bottom]
        elif line[i] != '-':
            return None
    if line[left] != '+':
        return None
    result = self.scan_up(top, left, bottom, right)
    if result is not None:
        rowseps = result
        return (rowseps, colseps)
    return None