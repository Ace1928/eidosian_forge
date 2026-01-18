from functools import singledispatch
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.errors import NumbaWarning
from numba.core.imputils import Registry
from numba.cuda import nvvmutils
from warnings import warn
@lower(print, types.VarArg(types.Any))
def print_varargs(context, builder, sig, args):
    """This function is a generic 'print' wrapper for arbitrary types.
    It dispatches to the appropriate 'print' implementations above
    depending on the detected real types in the signature."""
    vprint = nvvmutils.declare_vprint(builder.module)
    formats = []
    values = []
    for i, (argtype, argval) in enumerate(zip(sig.args, args)):
        argfmt, argvals = print_item(argtype, context, builder, argval)
        formats.append(argfmt)
        values.extend(argvals)
    rawfmt = ' '.join(formats) + '\n'
    if len(args) > 32:
        msg = 'CUDA print() cannot print more than 32 items. The raw format string will be emitted by the kernel instead.'
        warn(msg, NumbaWarning)
        rawfmt = rawfmt.replace('%', '%%')
    fmt = context.insert_string_const_addrspace(builder, rawfmt)
    array = cgutils.make_anonymous_struct(builder, values)
    arrayptr = cgutils.alloca_once_value(builder, array)
    vprint = nvvmutils.declare_vprint(builder.module)
    builder.call(vprint, (fmt, builder.bitcast(arrayptr, voidptr)))
    return context.get_dummy_value()