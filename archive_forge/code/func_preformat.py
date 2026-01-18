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
def preformat(self, text):
    """Format literal preformatted text."""
    text = self.escape(text.expandtabs())
    return replace(text, '\n\n', '\n \n', '\n\n', '\n \n', ' ', '&nbsp;', '\n', '<br>\n')