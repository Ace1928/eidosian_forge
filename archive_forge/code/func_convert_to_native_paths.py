import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def convert_to_native_paths(lst):
    """
    Converts a list of '/' separated paths into a list of
    native (os.sep separated) paths and converts to lowercase
    if the system is case insensitive.
    """
    newlst = []
    for i, rv in enumerate(lst):
        rv = os.path.join(*rv.split('/'))
        if sys.platform == 'win32':
            pos = rv.find(':')
            if pos != -1:
                if rv[pos + 1] != '\\':
                    rv = rv[:pos + 1] + '\\' + rv[pos + 1:]
        newlst.append(os.path.normcase(rv))
    return newlst