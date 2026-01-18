imported with ``from foo import ...`` was also updated.
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
import os
import sys
import traceback
import types
import weakref
import gc
import logging
from importlib import import_module, reload
from importlib.util import source_from_cache
def filename_and_mtime(self, module):
    if not hasattr(module, '__file__') or module.__file__ is None:
        return (None, None)
    if getattr(module, '__name__', None) in [None, '__mp_main__', '__main__']:
        return (None, None)
    filename = module.__file__
    path, ext = os.path.splitext(filename)
    if ext.lower() == '.py':
        py_filename = filename
    else:
        try:
            py_filename = source_from_cache(filename)
        except ValueError:
            return (None, None)
    try:
        pymtime = os.stat(py_filename).st_mtime
    except OSError:
        return (None, None)
    return (py_filename, pymtime)