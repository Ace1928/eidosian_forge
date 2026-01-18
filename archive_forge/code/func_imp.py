import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
def imp(context, builder, sig, args):
    instance_type = sig.args[0]
    if attr in instance_type.jit_methods:
        method = instance_type.jit_methods[attr]
    elif attr in instance_type.jit_static_methods:
        method = instance_type.jit_static_methods[attr]
        sig = sig.replace(args=sig.args[1:])
        args = args[1:]
    disp_type = types.Dispatcher(method)
    call = context.get_function(disp_type, sig)
    out = call(builder, args)
    _add_linking_libs(context, call)
    return imputils.impl_ret_new_ref(context, builder, sig.return_type, out)