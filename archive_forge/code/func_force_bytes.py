import inspect
import os
import sys
def force_bytes(s, encoding, errors='strict'):
    """Converts s to bytes, using the provided encoding. If s is already bytes,
    it is returned as is.

    If errors="strict" and s is bytes, its encoding is verified by decoding it;
    UnicodeError is raised if it cannot be decoded.
    """
    if isinstance(s, str):
        return s.encode(encoding, errors)
    else:
        s = bytes(s)
        if errors == 'strict':
            s.decode(encoding, errors)
        return s