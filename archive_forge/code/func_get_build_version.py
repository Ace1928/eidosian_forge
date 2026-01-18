import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def get_build_version():
    """Return the version of MSVC that was used to build Python.

    For Python 2.3 and up, the version number is included in
    sys.version.  For earlier versions, assume the compiler is MSVC 6.
    """
    prefix = 'MSC v.'
    i = sys.version.find(prefix)
    if i == -1:
        return 6
    i = i + len(prefix)
    s, rest = sys.version[i:].split(' ', 1)
    majorVersion = int(s[:-2]) - 6
    if majorVersion >= 13:
        majorVersion += 1
    minorVersion = int(s[2:3]) / 10.0
    if majorVersion == 6:
        minorVersion = 0
    if majorVersion >= 6:
        return majorVersion + minorVersion
    return None