import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def get_path_from_frame(frame, parent_path=None):
    """Return path of the module given a frame object from the call stack.

    Returned path is relative to parent_path when given,
    otherwise it is absolute path.
    """
    try:
        caller_file = eval('__file__', frame.f_globals, frame.f_locals)
        d = os.path.dirname(os.path.abspath(caller_file))
    except NameError:
        caller_name = eval('__name__', frame.f_globals, frame.f_locals)
        __import__(caller_name)
        mod = sys.modules[caller_name]
        if hasattr(mod, '__file__'):
            d = os.path.dirname(os.path.abspath(mod.__file__))
        else:
            d = os.path.abspath('.')
    if parent_path is not None:
        d = rel_path(d, parent_path)
    return d or '.'