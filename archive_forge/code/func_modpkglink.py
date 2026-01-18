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
def modpkglink(self, modpkginfo):
    """Make a link for a module or package to display in an index."""
    name, path, ispackage, shadowed = modpkginfo
    if shadowed:
        return self.grey(name)
    if path:
        url = '%s.%s.html' % (path, name)
    else:
        url = '%s.html' % name
    if ispackage:
        text = '<strong>%s</strong>&nbsp;(package)' % name
    else:
        text = name
    return '<a href="%s">%s</a>' % (url, text)