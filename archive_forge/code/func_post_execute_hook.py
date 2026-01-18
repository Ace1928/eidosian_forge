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
def post_execute_hook(self):
    """Cache the modification times of any modules imported in this execution"""
    newly_loaded_modules = set(sys.modules) - self.loaded_modules
    for modname in newly_loaded_modules:
        _, pymtime = self._reloader.filename_and_mtime(sys.modules[modname])
        if pymtime is not None:
            self._reloader.modules_mtimes[modname] = pymtime
    self.loaded_modules.update(newly_loaded_modules)