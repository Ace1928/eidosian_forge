import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def display_linize(msg, columns, blank_line=False):
    """Returns lines obtained by splitting msg over multiple lines.

    Warning: if msg is empty, returns an empty list of lines"""
    if not msg:
        return [''] if blank_line else []
    msg = fmtstr(msg)
    try:
        display_lines = list(msg.width_aware_splitlines(columns))
    except ValueError:
        display_lines = [msg[start:end] for start, end in zip(range(0, len(msg), columns), range(columns, len(msg) + columns, columns))]
    return display_lines