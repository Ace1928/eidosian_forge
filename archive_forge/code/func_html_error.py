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
def html_error(url, exc):
    heading = html.heading('<strong class="title">Error</strong>')
    contents = '<br>'.join((html.escape(line) for line in format_exception_only(type(exc), exc)))
    contents = heading + html.bigsection(url, 'error', contents)
    return ('Error - %s' % url, contents)