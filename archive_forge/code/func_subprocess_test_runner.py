import cmath
import contextlib
from collections import defaultdict
import enum
import gc
import math
import platform
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import io
import ctypes
import multiprocessing as mp
import warnings
import traceback
from contextlib import contextmanager
import uuid
import importlib
import types as pytypes
from functools import cached_property
import numpy as np
from numba import testing, types
from numba.core import errors, typing, utils, config, cpu
from numba.core.typing import cffi_utils
from numba.core.compiler import (compile_extra, Flags,
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
import unittest
from numba.core.runtime import rtsys
from numba.np import numpy_support
from numba.core.runtime import _nrt_python as _nrt
from numba.core.extending import (
from numba.core.datamodel.models import OpaqueModel
def subprocess_test_runner(self, test_module, test_class=None, test_name=None, envvars=None, timeout=60):
    """
        Runs named unit test(s) as specified in the arguments as:
        test_module.test_class.test_name. test_module must always be supplied
        and if no further refinement is made with test_class and test_name then
        all tests in the module will be run. The tests will be run in a
        subprocess with environment variables specified in `envvars`.
        If given, envvars must be a map of form:
            environment variable name (str) -> value (str)
        It is most convenient to use this method in conjunction with
        @needs_subprocess as the decorator will cause the decorated test to be
        skipped unless the `SUBPROC_TEST` environment variable is set to 1
        (this special environment variable is set by this method such that the
        specified test(s) will not be skipped in the subprocess).


        Following execution in the subprocess this method will check the test(s)
        executed without error. The timeout kwarg can be used to allow more time
        for longer running tests, it defaults to 60 seconds.
        """
    themod = self.__module__
    thecls = type(self).__name__
    parts = (test_module, test_class, test_name)
    fully_qualified_test = '.'.join((x for x in parts if x is not None))
    cmd = [sys.executable, '-m', 'numba.runtests', fully_qualified_test]
    env_copy = os.environ.copy()
    env_copy['SUBPROC_TEST'] = '1'
    try:
        env_copy['COVERAGE_PROCESS_START'] = os.environ['COVERAGE_RCFILE']
    except KeyError:
        pass
    envvars = pytypes.MappingProxyType({} if envvars is None else envvars)
    env_copy.update(envvars)
    status = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env_copy, universal_newlines=True)
    streams = f'\ncaptured stdout: {status.stdout}\ncaptured stderr: {status.stderr}'
    self.assertEqual(status.returncode, 0, streams)
    no_tests_ran = 'NO TESTS RAN'
    if no_tests_ran in status.stderr:
        self.skipTest(no_tests_ran)
    else:
        self.assertIn('OK', status.stderr)