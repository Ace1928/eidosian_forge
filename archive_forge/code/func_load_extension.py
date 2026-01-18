import os
import os.path
import sys
from importlib import import_module, reload
from traitlets.config.configurable import Configurable
from IPython.utils.path import ensure_dir_exists
from traitlets import Instance
def load_extension(self, module_str: str):
    """Load an IPython extension by its module name.

        Returns the string "already loaded" if the extension is already loaded,
        "no load function" if the module doesn't have a load_ipython_extension
        function, or None if it succeeded.
        """
    try:
        return self._load_extension(module_str)
    except ModuleNotFoundError:
        if module_str in BUILTINS_EXTS:
            BUILTINS_EXTS[module_str] = True
            return self._load_extension('IPython.extensions.' + module_str)
        raise