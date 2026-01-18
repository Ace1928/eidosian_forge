import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def structure_from_cells(self):
    colspecs = [end - start for start, end in self.columns]
    first_body_row = 0
    if self.head_body_sep:
        for i in range(len(self.table)):
            if self.table[i][0][2] > self.head_body_sep:
                first_body_row = i
                break
    return (colspecs, self.table[:first_body_row], self.table[first_body_row:])