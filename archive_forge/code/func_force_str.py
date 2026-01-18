import inspect
import os
import sys
def force_str(s, encoding, errors='strict'):
    """Converts s to str, using the provided encoding. If s is already str,
    it is returned as is.
    """
    return s.decode(encoding, errors) if isinstance(s, bytes) else str(s)