import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def read_values(base, key):
    """Return dict of registry keys and values.

    All names are converted to lowercase.
    """
    try:
        handle = RegOpenKeyEx(base, key)
    except RegError:
        return None
    d = {}
    i = 0
    while True:
        try:
            name, value, type = RegEnumValue(handle, i)
        except RegError:
            break
        name = name.lower()
        d[convert_mbcs(name)] = convert_mbcs(value)
        i += 1
    return d