import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def isnamedtuple(f):
    """Returns True if the argument is a namedtuple-like."""
    if not (tf_inspect.isclass(f) and issubclass(f, tuple)):
        return False
    if not hasattr(f, '_fields'):
        return False
    fields = getattr(f, '_fields')
    if not isinstance(fields, tuple):
        return False
    if not all((isinstance(f, str) for f in fields)):
        return False
    return True