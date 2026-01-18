from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
@property
def llvm_cpython_wrapper_name(self):
    """
        The LLVM-registered name for a CPython-compatible wrapper of the
        raw function (i.e. a PyCFunctionWithKeywords).
        """
    return itanium_mangler.prepend_namespace(self.mangled_name, ns='cpython')