import os
import shutil
import subprocess
import sys
def find_msvcrt():
    """Return the name of the VC runtime dll"""
    version = _get_build_version()
    if version is None:
        return None
    if version <= 6:
        clibname = 'msvcrt'
    elif version <= 13:
        clibname = 'msvcr%d' % (version * 10)
    else:
        return None
    import importlib.machinery
    if '_d.pyd' in importlib.machinery.EXTENSION_SUFFIXES:
        clibname += 'd'
    return clibname + '.dll'