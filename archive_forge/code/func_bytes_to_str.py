import inspect
import functools
import sys
import warnings
from eventlet.support import greenlets
def bytes_to_str(b, encoding='ascii'):
    return b.decode(encoding)