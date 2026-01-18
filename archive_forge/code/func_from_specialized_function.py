from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
@classmethod
def from_specialized_function(cls, func_ir, typemap, restype, calltypes, mangler, inline, noalias, abi_tags):
    """
        Build a FunctionDescriptor for a given specialization of a Python
        function (in nopython mode).
        """
    return cls._from_python_function(func_ir, typemap, restype, calltypes, native=True, mangler=mangler, inline=inline, noalias=noalias, abi_tags=abi_tags)