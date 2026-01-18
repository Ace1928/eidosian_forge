import re
import docutils
from docutils import nodes, writers, languages
def append_cell(self, cell_lines):
    """cell_lines is an array of lines"""
    start = 0
    if len(cell_lines) > 0 and cell_lines[0] == '.sp\n':
        start = 1
    self._rows[-1].append(cell_lines[start:])
    if len(self._coldefs) < len(self._rows[-1]):
        self._coldefs.append('l')