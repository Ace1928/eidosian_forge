import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def printf(builder, format, *args):
    """
    Calls printf().
    Argument `format` is expected to be a Python string.
    Values to be printed are listed in `args`.

    Note: There is no checking to ensure there is correct number of values
    in `args` and there type matches the declaration in the format string.
    """
    assert isinstance(format, str)
    mod = builder.module
    cstring = voidptr_t
    fmt_bytes = make_bytearray((format + '\x00').encode('ascii'))
    global_fmt = global_constant(mod, 'printf_format', fmt_bytes)
    fnty = ir.FunctionType(int32_t, [cstring], var_arg=True)
    try:
        fn = mod.get_global('printf')
    except KeyError:
        fn = ir.Function(mod, fnty, name='printf')
    ptr_fmt = builder.bitcast(global_fmt, cstring)
    return builder.call(fn, [ptr_fmt] + list(args))