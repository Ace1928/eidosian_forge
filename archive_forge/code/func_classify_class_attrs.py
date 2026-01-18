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
def classify_class_attrs(object):
    """Wrap inspect.classify_class_attrs, with fixup for data descriptors."""
    results = []
    for name, kind, cls, value in inspect.classify_class_attrs(object):
        if inspect.isdatadescriptor(value):
            kind = 'data descriptor'
            if isinstance(value, property) and value.fset is None:
                kind = 'readonly property'
        results.append((name, kind, cls, value))
    return results