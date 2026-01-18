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
@contextlib.contextmanager
def cleanup_object(self, test):
    """
        A context manager which cleans up unwanted attributes on a test case
        (or any other object).
        """
    vanilla_attrs = set(test.__dict__)
    try:
        yield test
    finally:
        spurious_attrs = set(test.__dict__) - vanilla_attrs
        for name in spurious_attrs:
            del test.__dict__[name]