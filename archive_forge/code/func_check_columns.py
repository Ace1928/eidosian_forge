import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def check_columns(self, lines, first_line, columns):
    """
        Check for text in column margins and text overflow in the last column.
        Raise TableMarkupError if anything but whitespace is in column margins.
        Adjust the end value for the last column if there is text overflow.
        """
    columns.append((sys.maxsize, None))
    lastcol = len(columns) - 2
    lines = [strip_combining_chars(line) for line in lines]
    for i in range(len(columns) - 1):
        start, end = columns[i]
        nextstart = columns[i + 1][0]
        offset = 0
        for line in lines:
            if i == lastcol and line[end:].strip():
                text = line[start:].rstrip()
                new_end = start + len(text)
                main_start, main_end = self.columns[-1]
                columns[i] = (start, max(main_end, new_end))
                if new_end > main_end:
                    self.columns[-1] = (main_start, new_end)
            elif line[end:nextstart].strip():
                raise TableMarkupError('Text in column margin in table line %s.' % (first_line + offset + 1), offset=first_line + offset)
            offset += 1
    columns.pop()