from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
def lookup_globals(self):
    """
        Return the global dictionary of the function.
        It may not match the Module's globals if the function is created
        dynamically (i.e. exec)
        """
    return self.global_dict or self.lookup_module().__dict__