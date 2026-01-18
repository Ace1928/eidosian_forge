import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
def check_linecache_ipython(*args):
    """Deprecated since IPython 8.6.  Call linecache.checkcache() directly.

    It was already not necessary to call this function directly.  If no
    CachingCompiler had been created, this function would fail badly.  If
    an instance had been created, this function would've been monkeypatched
    into place.

    As of IPython 8.6, the monkeypatching has gone away entirely.  But there
    were still internal callers of this function, so maybe external callers
    also existed?
    """
    import warnings
    warnings.warn('Deprecated Since IPython 8.6, Just call linecache.checkcache() directly.', DeprecationWarning, stacklevel=2)
    linecache.checkcache()