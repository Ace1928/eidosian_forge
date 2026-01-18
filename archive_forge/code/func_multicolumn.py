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
def multicolumn(self, list, format):
    """Format a list of items into a multi-column list."""
    result = ''
    rows = (len(list) + 3) // 4
    for col in range(4):
        result = result + '<td class="multicolumn">'
        for i in range(rows * col, rows * col + rows):
            if i < len(list):
                result = result + format(list[i]) + '<br>\n'
        result = result + '</td>'
    return '<table><tr>%s</tr></table>' % result