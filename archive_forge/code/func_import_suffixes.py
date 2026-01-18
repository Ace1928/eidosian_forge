import os, glob, re, sys
from distutils import sysconfig
def import_suffixes():
    if sys.version_info >= (3, 4):
        import importlib.machinery
        return importlib.machinery.EXTENSION_SUFFIXES
    else:
        import imp
        result = []
        for t in imp.get_suffixes():
            result.append(t[0])
        return result