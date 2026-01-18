import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def read_keys(base, key):
    """Return list of registry keys."""
    try:
        handle = RegOpenKeyEx(base, key)
    except RegError:
        return None
    L = []
    i = 0
    while True:
        try:
            k = RegEnumKey(handle, i)
        except RegError:
            break
        L.append(k)
        i += 1
    return L