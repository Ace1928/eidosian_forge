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
def docdata(self, object, name=None, mod=None, cl=None):
    """Produce text documentation for a data descriptor."""
    results = []
    push = results.append
    if name:
        push(self.bold(name))
        push('\n')
    doc = getdoc(object) or ''
    if doc:
        push(self.indent(doc))
        push('\n')
    return ''.join(results)