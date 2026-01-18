from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
@classmethod
def from_object_mode_function(cls, func_ir):
    """
        Build a FunctionDescriptor for an object mode variant of a Python
        function.
        """
    typemap = defaultdict(lambda: types.pyobject)
    calltypes = typemap.copy()
    restype = types.pyobject
    return cls._from_python_function(func_ir, typemap, restype, calltypes, native=False)