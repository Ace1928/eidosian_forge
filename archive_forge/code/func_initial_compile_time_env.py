from __future__ import absolute_import
import cython
import os
import platform
from unicodedata import normalize
from contextlib import contextmanager
from .. import Utils
from ..Plex.Scanners import Scanner
from ..Plex.Errors import UnrecognizedInput
from .Errors import error, warning, hold_errors, release_errors, CompileError
from .Lexicon import any_string_prefix, make_lexicon, IDENT
from .Future import print_function
def initial_compile_time_env():
    benv = CompileTimeScope()
    names = ('UNAME_SYSNAME', 'UNAME_NODENAME', 'UNAME_RELEASE', 'UNAME_VERSION', 'UNAME_MACHINE')
    for name, value in zip(names, platform.uname()):
        benv.declare(name, value)
    try:
        import __builtin__ as builtins
    except ImportError:
        import builtins
    names = ('False', 'True', 'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 'chr', 'cmp', 'complex', 'dict', 'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset', 'hash', 'hex', 'int', 'len', 'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'zip')
    for name in names:
        try:
            benv.declare(name, getattr(builtins, name))
        except AttributeError:
            pass
    from functools import reduce
    benv.declare('reduce', reduce)
    benv.declare('unicode', getattr(builtins, 'unicode', getattr(builtins, 'str')))
    benv.declare('long', getattr(builtins, 'long', getattr(builtins, 'int')))
    benv.declare('xrange', getattr(builtins, 'xrange', getattr(builtins, 'range')))
    denv = CompileTimeScope(benv)
    return denv