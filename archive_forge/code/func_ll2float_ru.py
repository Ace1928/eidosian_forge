import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def ll2float_ru(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('int64'),): ('__nv_ll2float_ru', core.dtype('fp32'))}, is_pure=True, _builder=_builder)