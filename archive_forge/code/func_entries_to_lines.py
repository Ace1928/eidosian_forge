import difflib
import patiencediff
from merge3 import Merge3
from ... import debug, merge, osutils
from ...trace import mutter
def entries_to_lines(entries):
    """Turn a list of entries into a flat iterable of lines."""
    for entry in entries:
        yield from entry