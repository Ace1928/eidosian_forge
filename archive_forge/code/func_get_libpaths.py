import re
from os import environ, path
from sys import executable
from ctypes import c_void_p, sizeof
from subprocess import Popen, PIPE, DEVNULL
from sys import maxsize
def get_libpaths():
    """
    On AIX, the buildtime searchpath is stored in the executable.
    as "loader header information".
    The command /usr/bin/dump -H extracts this info.
    Prefix searched libraries with LD_LIBRARY_PATH (preferred),
    or LIBPATH if defined. These paths are appended to the paths
    to libraries the python executable is linked with.
    This mimics AIX dlopen() behavior.
    """
    libpaths = environ.get('LD_LIBRARY_PATH')
    if libpaths is None:
        libpaths = environ.get('LIBPATH')
    if libpaths is None:
        libpaths = []
    else:
        libpaths = libpaths.split(':')
    objects = get_ld_headers(executable)
    for _, lines in objects:
        for line in lines:
            path = line.split()[1]
            if '/' in path:
                libpaths.extend(path.split(':'))
    return libpaths