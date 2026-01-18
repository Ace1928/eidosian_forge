import itertools
import re
from collections import deque
from contextlib import contextmanager
from sqlparse.compat import text_type
def remove_quotes(val):
    """Helper that removes surrounding quotes from strings."""
    if val is None:
        return
    if val[0] in ('"', "'") and val[0] == val[-1]:
        val = val[1:-1]
    return val