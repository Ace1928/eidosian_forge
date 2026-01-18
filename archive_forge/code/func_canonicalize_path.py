import logging
import os
import time
def canonicalize_path(path):
    """Canonicalizes a potential path.

    Returns a binary string encoded into filesystem encoding.
    """
    if isinstance(path, bytes):
        return path
    if isinstance(path, str):
        return os.fsencode(path)
    else:
        return canonicalize_path(str(path))