from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
@property
def llvm_func_name(self):
    """
        The LLVM-registered name for the raw function.
        """
    return self.mangled_name