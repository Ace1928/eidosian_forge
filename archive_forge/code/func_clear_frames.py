import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def clear_frames(tb):
    """Clear all references to local variables in the frames of a traceback."""
    while tb is not None:
        try:
            tb.tb_frame.clear()
        except RuntimeError:
            pass
        tb = tb.tb_next