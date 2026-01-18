import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def getdocloc(self, object, basedir=sysconfig.get_path('stdlib')):
    """Return the location of module docs or None"""
    try:
        file = inspect.getabsfile(object)
    except TypeError:
        file = '(built-in)'
    docloc = os.environ.get('PYTHONDOCS', self.PYTHONDOCS)
    basedir = os.path.normcase(basedir)
    if isinstance(object, type(os)) and (object.__name__ in ('errno', 'exceptions', 'gc', 'imp', 'marshal', 'posix', 'signal', 'sys', '_thread', 'zipimport') or (file.startswith(basedir) and (not file.startswith(os.path.join(basedir, 'site-packages'))))) and (object.__name__ not in ('xml.etree', 'test.pydoc_mod')):
        if docloc.startswith(('http://', 'https://')):
            docloc = '{}/{}.html'.format(docloc.rstrip('/'), object.__name__.lower())
        else:
            docloc = os.path.join(docloc, object.__name__.lower() + '.html')
    else:
        docloc = None
    return docloc