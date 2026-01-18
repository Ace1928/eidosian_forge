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
def mark_module_reloadable(self, module_name):
    """Reload the named module in the future (if it is imported)"""
    try:
        del self.skip_modules[module_name]
    except KeyError:
        pass
    self.modules[module_name] = True