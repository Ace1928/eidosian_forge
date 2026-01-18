from __future__ import annotations
import re
from fractions import Fraction
def stream_has_colors(stream):
    """True if stream supports colors. Python cookbook, #475186."""
    if not hasattr(stream, 'isatty'):
        return False
    if not stream.isatty():
        return False
    try:
        import curses
        curses.setupterm()
    except Exception:
        return False
    else:
        return curses.tigetnum('colors') > 2