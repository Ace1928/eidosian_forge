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
def aimport_module(self, module_name):
    """Import a module, and mark it reloadable

        Returns
        -------
        top_module : module
            The imported module if it is top-level, or the top-level
        top_name : module
            Name of top_module

        """
    self.mark_module_reloadable(module_name)
    import_module(module_name)
    top_name = module_name.split('.')[0]
    top_module = sys.modules[top_name]
    return (top_module, top_name)