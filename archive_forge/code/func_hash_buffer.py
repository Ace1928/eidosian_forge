from __future__ import annotations
import binascii
import hashlib
def hash_buffer(buf, hasher=None):
    """
    Hash a bytes-like (buffer-compatible) object.  This function returns
    a good quality hash but is not cryptographically secure.  The fastest
    available algorithm is selected.  A fixed-length bytes object is returned.
    """
    if hasher is not None:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            pass
    for hasher in hashers:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            pass
    raise TypeError(f'unsupported type for hashing: {type(buf)}')