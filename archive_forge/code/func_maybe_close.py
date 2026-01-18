import sys
import marshal
import contextlib
import dis
from . import _imp
from ._imp import find_module, PY_COMPILED, PY_FROZEN, PY_SOURCE
from .extern.packaging.version import Version
def maybe_close(f):

    @contextlib.contextmanager
    def empty():
        yield
        return
    if not f:
        return empty()
    return contextlib.closing(f)