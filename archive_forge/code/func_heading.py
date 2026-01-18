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
def heading(self, title, extras=''):
    """Format a page heading."""
    return '\n<table class="heading">\n<tr class="heading-text decor">\n<td class="title">&nbsp;<br>%s</td>\n<td class="extra">%s</td></tr></table>\n    ' % (title, extras or '&nbsp;')