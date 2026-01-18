from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def run_doctest(module, verbosity=None, optionflags=0):
    """Run doctest on the given module.  Return (#failures, #tests).

    If optional argument verbosity is not specified (or is None), pass
    support's belief about verbosity on to doctest.  Else doctest's
    usual behavior is used (it searches sys.argv for -v).
    """
    import doctest
    if verbosity is None:
        verbosity = verbose
    else:
        verbosity = None
    f, t = doctest.testmod(module, verbose=verbosity, optionflags=optionflags)
    if f:
        raise TestFailed('%d of %d doctests failed' % (f, t))
    if verbose:
        print('doctest (%s) ... %d tests with zero failures' % (module.__name__, t))
    return (f, t)