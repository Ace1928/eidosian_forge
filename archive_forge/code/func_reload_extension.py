import os
import os.path
import sys
from importlib import import_module, reload
from traitlets.config.configurable import Configurable
from IPython.utils.path import ensure_dir_exists
from traitlets import Instance
def reload_extension(self, module_str: str):
    """Reload an IPython extension by calling reload.

        If the module has not been loaded before,
        :meth:`InteractiveShell.load_extension` is called. Otherwise
        :func:`reload` is called and then the :func:`load_ipython_extension`
        function of the module, if it exists is called.
        """
    if BUILTINS_EXTS.get(module_str, False) is True:
        module_str = 'IPython.extensions.' + module_str
    if module_str in self.loaded and module_str in sys.modules:
        self.unload_extension(module_str)
        mod = sys.modules[module_str]
        reload(mod)
        if self._call_load_ipython_extension(mod):
            self.loaded.add(module_str)
    else:
        self.load_extension(module_str)