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
def html_search(key):
    """Search results page."""
    search_result = []

    def callback(path, modname, desc):
        if modname[-9:] == '.__init__':
            modname = modname[:-9] + ' (package)'
        search_result.append((modname, desc and '- ' + desc))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        def onerror(modname):
            pass
        ModuleScanner().run(callback, key, onerror=onerror)

    def bltinlink(name):
        return '<a href="%s.html">%s</a>' % (name, name)
    results = []
    heading = html.heading('<strong class="title">Search Results</strong>')
    for name, desc in search_result:
        results.append(bltinlink(name) + desc)
    contents = heading + html.bigsection('key = %s' % key, 'index', '<br>'.join(results))
    return ('Search Results', contents)