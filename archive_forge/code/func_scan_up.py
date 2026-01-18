import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def scan_up(self, top, left, bottom, right):
    """
        Noting row boundaries, see if we can return to the starting point.
        """
    rowseps = {}
    for i in range(bottom - 1, top, -1):
        if self.block[i][left] == '+':
            rowseps[i] = [left]
        elif self.block[i][left] != '|':
            return None
    return rowseps