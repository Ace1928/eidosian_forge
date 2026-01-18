import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def getimmediatesource(obj):
    """A variant of inspect.getsource that ignores the __wrapped__ property."""
    with _linecache_lock:
        _fix_linecache_record(obj)
        lines, lnum = inspect.findsource(obj)
        return ''.join(inspect.getblock(lines[lnum:]))