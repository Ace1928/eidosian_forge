import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def clz(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('int32'),): ('__nv_clz', core.dtype('int32')), (core.dtype('int64'),): ('__nv_clzll', core.dtype('int32'))}, is_pure=True, _builder=_builder)