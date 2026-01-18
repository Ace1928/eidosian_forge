from distutils.dir_util import remove_tree, mkpath
from distutils import log
from types import CodeType
import sys
import os
import re
import textwrap
import marshal
from setuptools.extension import Library
from setuptools import Command
from .._path import ensure_directory
from sysconfig import get_path, get_python_version
def scan_module(egg_dir, base, name, stubs):
    """Check whether module possibly uses unsafe-for-zipfile stuff"""
    filename = os.path.join(base, name)
    if filename[:-1] in stubs:
        return True
    pkg = base[len(egg_dir) + 1:].replace(os.sep, '.')
    module = pkg + (pkg and '.' or '') + os.path.splitext(name)[0]
    if sys.version_info < (3, 7):
        skip = 12
    else:
        skip = 16
    f = open(filename, 'rb')
    f.read(skip)
    code = marshal.load(f)
    f.close()
    safe = True
    symbols = dict.fromkeys(iter_symbols(code))
    for bad in ['__file__', '__path__']:
        if bad in symbols:
            log.warn('%s: module references %s', module, bad)
            safe = False
    if 'inspect' in symbols:
        for bad in ['getsource', 'getabsfile', 'getsourcefile', 'getfilegetsourcelines', 'findsource', 'getcomments', 'getframeinfo', 'getinnerframes', 'getouterframes', 'stack', 'trace']:
            if bad in symbols:
                log.warn('%s: module MAY be using inspect.%s', module, bad)
                safe = False
    return safe