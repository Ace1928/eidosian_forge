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
def classlink(self, object, modname):
    """Make a link for a class."""
    name, module = (object.__name__, sys.modules.get(object.__module__))
    if hasattr(module, name) and getattr(module, name) is object:
        return '<a href="%s.html#%s">%s</a>' % (module.__name__, name, classname(object, modname))
    return classname(object, modname)