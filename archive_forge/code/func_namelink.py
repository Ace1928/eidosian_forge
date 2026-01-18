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
def namelink(self, name, *dicts):
    """Make a link for an identifier, given name-to-URL mappings."""
    for dict in dicts:
        if name in dict:
            return '<a href="%s">%s</a>' % (dict[name], name)
    return name