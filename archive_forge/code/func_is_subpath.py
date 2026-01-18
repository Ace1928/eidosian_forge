import asyncio
import fnmatch
import logging
import os
import sys
import types
import warnings
from contextlib import contextmanager
from bokeh.application.handlers import CodeHandler
from ..util import fullpath
from .state import state
def is_subpath(subpath, path):
    try:
        return os.path.commonpath([path, subpath]) == path
    except Exception:
        return False