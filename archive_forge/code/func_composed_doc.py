from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def composed_doc(*fs):
    """Generate a docstring for the composition of fs.
            """
    if not fs:
        return '*args, **kwargs'
    return '{f}({g})'.format(f=fs[0].__name__, g=composed_doc(*fs[1:]))