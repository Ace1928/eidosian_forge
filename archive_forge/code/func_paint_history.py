import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def paint_history(rows, columns, display_lines):
    lines = []
    for r, line in zip(range(rows), display_lines[-rows:]):
        lines.append(fmtstr(line[:columns]))
    r = fsarray(lines, width=columns)
    assert r.shape[0] <= rows, repr(r.shape) + ' ' + repr(rows)
    assert r.shape[1] <= columns, repr(r.shape) + ' ' + repr(columns)
    return r