import collections
from .objects import ZERO_SHA, format_timezone, parse_timezone
def read_reflog(f):
    """Read reflog.

    Args:
      f: File-like object
    Returns: Iterator over Entry objects
    """
    for line in f:
        yield parse_reflog_line(line)