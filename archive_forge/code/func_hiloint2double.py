import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def hiloint2double(arg0, arg1, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0, arg1], {(core.dtype('int32'), core.dtype('int32')): ('__nv_hiloint2double', core.dtype('fp64'))}, is_pure=True, _builder=_builder)