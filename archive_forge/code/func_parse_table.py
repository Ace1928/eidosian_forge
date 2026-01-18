import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def parse_table(self):
    """
        First determine the column boundaries from the top border, then
        process rows.  Each row may consist of multiple lines; accumulate
        lines until a row is complete.  Call `self.parse_row` to finish the
        job.
        """
    self.columns = self.parse_columns(self.block[0], 0)
    self.border_end = self.columns[-1][1]
    firststart, firstend = self.columns[0]
    offset = 1
    start = 1
    text_found = None
    while offset < len(self.block):
        line = self.block[offset]
        if self.span_pat.match(line):
            self.parse_row(self.block[start:offset], start, (line.rstrip(), offset))
            start = offset + 1
            text_found = None
        elif line[firststart:firstend].strip():
            if text_found and offset != start:
                self.parse_row(self.block[start:offset], start)
            start = offset
            text_found = 1
        elif not text_found:
            start = offset + 1
        offset += 1