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
def get_test_files(self, dir, pat='*.py', init_only=True):
    """
        Returns the list of \\*.py files (default) from which docstrings
        will be tested which are at or below directory ``dir``. By default,
        only those that have an __init__.py in their parent directory
        and do not start with ``test_`` will be included.
        """

    def importable(x):
        """
            Checks if given pathname x is an importable module by checking for
            __init__.py file.

            Returns True/False.

            Currently we only test if the __init__.py file exists in the
            directory with the file "x" (in theory we should also test all the
            parent dirs).
            """
        init_py = os.path.join(os.path.dirname(x), '__init__.py')
        return os.path.exists(init_py)
    dir = os.path.join(self._root_dir, convert_to_native_paths([dir])[0])
    g = []
    for path, folders, files in os.walk(dir):
        g.extend([os.path.join(path, f) for f in files if not f.startswith('test_') and fnmatch(f, pat)])
    if init_only:
        g = [x for x in g if importable(x)]
    return [os.path.normcase(gi) for gi in g]