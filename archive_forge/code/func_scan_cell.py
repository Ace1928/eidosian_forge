import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def scan_cell(self, top, left):
    """Starting at the top-left corner, start tracing out a cell."""
    assert self.block[top][left] == '+'
    result = self.scan_right(top, left)
    return result