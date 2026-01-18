import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def mulhi(arg0, arg1, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0, arg1], {(core.dtype('int32'), core.dtype('int32')): ('__nv_mulhi', core.dtype('int32')), (core.dtype('uint32'), core.dtype('uint32')): ('__nv_umulhi', core.dtype('uint32')), (core.dtype('int64'), core.dtype('int64')): ('__nv_mul64hi', core.dtype('int64')), (core.dtype('uint64'), core.dtype('uint64')): ('__nv_umul64hi', core.dtype('uint64'))}, is_pure=True, _builder=_builder)