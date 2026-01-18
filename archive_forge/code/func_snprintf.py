import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def snprintf(builder, buffer, bufsz, format, *args):
    """Calls libc snprintf(buffer, bufsz, format, ...args)
    """
    assert isinstance(format, str)
    mod = builder.module
    cstring = voidptr_t
    fmt_bytes = make_bytearray((format + '\x00').encode('ascii'))
    global_fmt = global_constant(mod, 'snprintf_format', fmt_bytes)
    fnty = ir.FunctionType(int32_t, [cstring, intp_t, cstring], var_arg=True)
    symbol = 'snprintf'
    if config.IS_WIN32:
        symbol = '_' + symbol
    try:
        fn = mod.get_global(symbol)
    except KeyError:
        fn = ir.Function(mod, fnty, name=symbol)
    ptr_fmt = builder.bitcast(global_fmt, cstring)
    return builder.call(fn, [buffer, bufsz, ptr_fmt] + list(args))