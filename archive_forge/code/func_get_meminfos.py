import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def get_meminfos(self, builder, ty, val):
    """Return a list of *(type, meminfo)* inside the given value.
        """
    datamodel = self._context.data_model_manager[ty]
    members = datamodel.traverse(builder)
    meminfos = []
    if datamodel.has_nrt_meminfo():
        mi = datamodel.get_nrt_meminfo(builder, val)
        meminfos.append((ty, mi))
    for mtyp, getter in members:
        field = getter(val)
        inner_meminfos = self.get_meminfos(builder, mtyp, field)
        meminfos.extend(inner_meminfos)
    return meminfos