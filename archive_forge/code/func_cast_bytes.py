import warnings
def cast_bytes(s, encoding='utf8', errors='strict'):
    """cast unicode or bytes to bytes"""
    warnings.warn('zmq.utils.strtypes is deprecated in pyzmq 23.', DeprecationWarning, stacklevel=2)
    if isinstance(s, bytes):
        return s
    elif isinstance(s, str):
        return s.encode(encoding, errors)
    else:
        raise TypeError('Expected unicode or bytes, got %r' % s)