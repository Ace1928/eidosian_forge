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
def run_in_subprocess_with_hash_randomization(function, function_args=(), function_kwargs=None, command=sys.executable, module='sympy.testing.runtests', force=False):
    """
    Run a function in a Python subprocess with hash randomization enabled.

    If hash randomization is not supported by the version of Python given, it
    returns False.  Otherwise, it returns the exit value of the command.  The
    function is passed to sys.exit(), so the return value of the function will
    be the return value.

    The environment variable PYTHONHASHSEED is used to seed Python's hash
    randomization.  If it is set, this function will return False, because
    starting a new subprocess is unnecessary in that case.  If it is not set,
    one is set at random, and the tests are run.  Note that if this
    environment variable is set when Python starts, hash randomization is
    automatically enabled.  To force a subprocess to be created even if
    PYTHONHASHSEED is set, pass ``force=True``.  This flag will not force a
    subprocess in Python versions that do not support hash randomization (see
    below), because those versions of Python do not support the ``-R`` flag.

    ``function`` should be a string name of a function that is importable from
    the module ``module``, like "_test".  The default for ``module`` is
    "sympy.testing.runtests".  ``function_args`` and ``function_kwargs``
    should be a repr-able tuple and dict, respectively.  The default Python
    command is sys.executable, which is the currently running Python command.

    This function is necessary because the seed for hash randomization must be
    set by the environment variable before Python starts.  Hence, in order to
    use a predetermined seed for tests, we must start Python in a separate
    subprocess.

    Hash randomization was added in the minor Python versions 2.6.8, 2.7.3,
    3.1.5, and 3.2.3, and is enabled by default in all Python versions after
    and including 3.3.0.

    Examples
    ========

    >>> from sympy.testing.runtests import (
    ... run_in_subprocess_with_hash_randomization)
    >>> # run the core tests in verbose mode
    >>> run_in_subprocess_with_hash_randomization("_test",
    ... function_args=("core",),
    ... function_kwargs={'verbose': True}) # doctest: +SKIP
    # Will return 0 if sys.executable supports hash randomization and tests
    # pass, 1 if they fail, and False if it does not support hash
    # randomization.

    """
    cwd = get_sympy_dir()
    p = subprocess.Popen([command, '-RV'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    p.communicate()
    if p.returncode != 0:
        return False
    hash_seed = os.getenv('PYTHONHASHSEED')
    if not hash_seed:
        os.environ['PYTHONHASHSEED'] = str(random.randrange(2 ** 32))
    elif not force:
        return False
    function_kwargs = function_kwargs or {}
    commandstring = 'import sys; from %s import %s;sys.exit(%s(*%s, **%s))' % (module, function, function, repr(function_args), repr(function_kwargs))
    try:
        p = subprocess.Popen([command, '-R', '-c', commandstring], cwd=cwd)
        p.communicate()
    except KeyboardInterrupt:
        p.wait()
    finally:
        if hash_seed is None:
            del os.environ['PYTHONHASHSEED']
        else:
            os.environ['PYTHONHASHSEED'] = hash_seed
        return p.returncode