import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def fast_cosf(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp32'),): ('__nv_fast_cosf', core.dtype('fp32'))}, is_pure=True, _builder=_builder)