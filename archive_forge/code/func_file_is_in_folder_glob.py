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
def file_is_in_folder_glob(filepath, folderpath_glob):
    """
    Test whether a file is in some folder with globbing support.

    Parameters
    ----------
    filepath : str
        A file path.
    folderpath_glob: str
        A path to a folder that may include globbing.
    """
    if not folderpath_glob.endswith('*'):
        if folderpath_glob.endswith('/'):
            folderpath_glob += '*'
        else:
            folderpath_glob += '/*'
    file_dir = os.path.dirname(filepath) + '/'
    return fnmatch.fnmatch(file_dir, folderpath_glob)