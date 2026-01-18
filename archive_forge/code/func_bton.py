import os
import platform
def bton(b, encoding='ISO-8859-1'):
    """Return the byte string as native string in the given encoding."""
    return b.decode(encoding)