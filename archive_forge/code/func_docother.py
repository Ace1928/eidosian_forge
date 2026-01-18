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
def docother(self, object, name=None, mod=None, parent=None, maxlen=None, doc=None):
    """Produce text documentation for a data object."""
    repr = self.repr(object)
    if maxlen:
        line = (name and name + ' = ' or '') + repr
        chop = maxlen - len(line)
        if chop < 0:
            repr = repr[:chop] + '...'
    line = (name and self.bold(name) + ' = ' or '') + repr
    if not doc:
        doc = getdoc(object)
    if doc:
        line += '\n' + self.indent(str(doc)) + '\n'
    return line