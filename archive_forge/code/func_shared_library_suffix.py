import os, glob, re, sys
from distutils import sysconfig
def shared_library_suffix():
    if sys.platform == 'win32':
        return 'lib'
    elif sys.platform == 'darwin':
        return 'dylib'
    else:
        return 'so.*'