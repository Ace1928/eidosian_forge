import os
import platform
def ntou(n, encoding='ISO-8859-1'):
    """Return the native string as Unicode with the given encoding."""
    assert_native(n)
    return n