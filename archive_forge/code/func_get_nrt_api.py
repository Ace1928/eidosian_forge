import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def get_nrt_api(self, builder):
    """Calls NRT_get_api(), which returns the NRT API function table.
        """
    self._require_nrt()
    fnty = ir.FunctionType(cgutils.voidptr_t, ())
    mod = builder.module
    fn = cgutils.get_or_insert_function(mod, fnty, 'NRT_get_api')
    return builder.call(fn, ())