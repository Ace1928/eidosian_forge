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
def splitdoc(doc):
    """Split a doc string into a synopsis line (if any) and the rest."""
    lines = doc.strip().split('\n')
    if len(lines) == 1:
        return (lines[0], '')
    elif len(lines) >= 2 and (not lines[1].rstrip()):
        return (lines[0], '\n'.join(lines[2:]))
    return ('', '\n'.join(lines))