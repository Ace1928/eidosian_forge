import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings
import numpy
import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
@contextlib.contextmanager
def assert_function_is_called(*args, times_called=1, **kwargs):
    """A handy wrapper for unittest.mock to check if a function is called.

    Args:
        *args: Arguments of `mock.patch`.
        times_called (int): The number of times the function should be
            called. Default is ``1``.
        **kwargs: Keyword arguments of `mock.patch`.

    """
    with mock.patch(*args, **kwargs) as handle:
        yield
        assert handle.call_count == times_called