import atexit
import contextlib
import functools
import importlib
import inspect
import os
import os.path as op
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from subprocess import STDOUT, CalledProcessError, TimeoutExpired, check_output
from unittest import TestCase
import joblib
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose as np_assert_allclose
from numpy.testing import (
import sklearn
from sklearn.utils import (
from sklearn.utils._array_api import _check_array_api_dispatch
from sklearn.utils.fixes import VisibleDeprecationWarning, parse_version, sp_version
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
def assert_run_python_script_without_output(source_code, pattern='.+', timeout=60):
    """Utility to check assertions in an independent Python subprocess.

    The script provided in the source code should return 0 and the stdtout +
    stderr should not match the pattern `pattern`.

    This is a port from cloudpickle https://github.com/cloudpipe/cloudpickle

    Parameters
    ----------
    source_code : str
        The Python source code to execute.
    pattern : str
        Pattern that the stdout + stderr should not match. By default, unless
        stdout + stderr are both empty, an error will be raised.
    timeout : int, default=60
        Time in seconds before timeout.
    """
    fd, source_file = tempfile.mkstemp(suffix='_src_test_sklearn.py')
    os.close(fd)
    try:
        with open(source_file, 'wb') as f:
            f.write(source_code.encode('utf-8'))
        cmd = [sys.executable, source_file]
        cwd = op.normpath(op.join(op.dirname(sklearn.__file__), '..'))
        env = os.environ.copy()
        try:
            env['PYTHONPATH'] = os.pathsep.join([cwd, env['PYTHONPATH']])
        except KeyError:
            env['PYTHONPATH'] = cwd
        kwargs = {'cwd': cwd, 'stderr': STDOUT, 'env': env}
        coverage_rc = os.environ.get('COVERAGE_PROCESS_START')
        if coverage_rc:
            kwargs['env']['COVERAGE_PROCESS_START'] = coverage_rc
        kwargs['timeout'] = timeout
        try:
            try:
                out = check_output(cmd, **kwargs)
            except CalledProcessError as e:
                raise RuntimeError('script errored with output:\n%s' % e.output.decode('utf-8'))
            out = out.decode('utf-8')
            if re.search(pattern, out):
                if pattern == '.+':
                    expectation = 'Expected no output'
                else:
                    expectation = f'The output was not supposed to match {pattern!r}'
                message = f'{expectation}, got the following output instead: {out!r}'
                raise AssertionError(message)
        except TimeoutExpired as e:
            raise RuntimeError('script timeout, output so far:\n%s' % e.output.decode('utf-8'))
    finally:
        os.unlink(source_file)