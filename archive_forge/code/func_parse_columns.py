import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def parse_columns(self, line, offset):
    """
        Given a column span underline, return a list of (begin, end) pairs.
        """
    cols = []
    end = 0
    while True:
        begin = line.find('-', end)
        end = line.find(' ', begin)
        if begin < 0:
            break
        if end < 0:
            end = len(line)
        cols.append((begin, end))
    if self.columns:
        if cols[-1][1] != self.border_end:
            raise TableMarkupError('Column span incomplete in table line %s.' % (offset + 1), offset=offset)
        cols[-1] = (cols[-1][0], self.columns[-1][1])
    return cols