from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
def lookup_module(self):
    """
        Return the module in which this function is supposed to exist.
        This may be a dummy module if the function was dynamically
        generated or the module can't be found.
        """
    if self.modname == _dynamic_modname:
        return _dynamic_module
    else:
        try:
            return importlib.import_module(self.modname)
        except ImportError:
            return _dynamic_module