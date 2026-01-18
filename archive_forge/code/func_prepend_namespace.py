import re
from numba.core import types
def prepend_namespace(mangled, ns):
    """
    Prepend namespace to mangled name.
    """
    if not mangled.startswith(PREFIX):
        raise ValueError('input is not a mangled name')
    elif mangled.startswith(PREFIX + 'N'):
        remaining = mangled[3:]
        ret = PREFIX + 'N' + mangle_identifier(ns) + remaining
    else:
        remaining = mangled[2:]
        head, tail = _split_mangled_ident(remaining)
        ret = PREFIX + 'N' + mangle_identifier(ns) + head + 'E' + tail
    return ret