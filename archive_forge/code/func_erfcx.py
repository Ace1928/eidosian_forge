import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def erfcx(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp32'),): ('__nv_erfcxf', core.dtype('fp32')), (core.dtype('fp64'),): ('__nv_erfcx', core.dtype('fp64'))}, is_pure=True, _builder=_builder)