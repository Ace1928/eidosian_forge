import copy
import os
import sys
from importlib import import_module
from importlib.util import find_spec as importlib_find
def module_dir(module):
    """
    Find the name of the directory that contains a module, if possible.

    Raise ValueError otherwise, e.g. for namespace packages that are split
    over several directories.
    """
    paths = list(getattr(module, '__path__', []))
    if len(paths) == 1:
        return paths[0]
    else:
        filename = getattr(module, '__file__', None)
        if filename is not None:
            return os.path.dirname(filename)
    raise ValueError('Cannot determine directory containing %s' % module)