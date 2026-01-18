import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def int2float_rd(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('int32'),): ('__nv_int2float_rd', core.dtype('fp32'))}, is_pure=True, _builder=_builder)