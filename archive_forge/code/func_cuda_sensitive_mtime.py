import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import unittest
import warnings
import zlib
from functools import lru_cache
from io import StringIO
from unittest import result, runner, signals, suite, loader, case
from .loader import TestLoader
from numba.core import config
def cuda_sensitive_mtime(x):
    """
    Return a key for sorting tests bases on mtime and test name. For CUDA
    tests, interleaving tests from different classes is dangerous as the CUDA
    context might get reset unexpectedly between methods of a class, so for
    CUDA tests the key prioritises the test module and class ahead of the
    mtime.
    """
    cls = x.__class__
    key = _get_mtime(cls) + str(x)
    from numba.cuda.testing import CUDATestCase
    if CUDATestCase in cls.mro():
        key = '%s.%s %s' % (str(cls.__module__), str(cls.__name__), key)
    return key