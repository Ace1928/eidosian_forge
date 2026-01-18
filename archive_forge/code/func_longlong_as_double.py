import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def longlong_as_double(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('int64'),): ('__nv_longlong_as_double', core.dtype('fp64'))}, is_pure=True, _builder=_builder)