from typing import Any
def cast_int_addr(n: Any) -> int:
    """Cast an address to a Python int

    This could be a Python integer or a CFFI pointer
    """
    if isinstance(n, int):
        return n
    try:
        import cffi
    except ImportError:
        pass
    else:
        ffi = cffi.FFI()
        if isinstance(n, ffi.CData):
            return int(ffi.cast('size_t', n))
    raise ValueError('Cannot cast %r to int' % n)