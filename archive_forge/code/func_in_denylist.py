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
def in_denylist(filepath):
    return any((file_is_in_folder_glob(filepath, denylisted_folder) for denylisted_folder in DEFAULT_FOLDER_DENYLIST))