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
def listsymbols(self):
    self.output.write('\nHere is a list of the punctuation symbols which Python assigns special meaning\nto. Enter any symbol to get more help.\n\n')
    self.list(self.symbols.keys())