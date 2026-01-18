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
def get_sympy_dir():
    """
    Returns the root SymPy directory and set the global value
    indicating whether the system is case sensitive or not.
    """
    this_file = os.path.abspath(__file__)
    sympy_dir = os.path.join(os.path.dirname(this_file), '..', '..')
    sympy_dir = os.path.normpath(sympy_dir)
    return os.path.normcase(sympy_dir)