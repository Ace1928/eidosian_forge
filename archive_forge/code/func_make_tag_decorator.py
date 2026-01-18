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
def make_tag_decorator(known_tags):
    """
    Create a decorator allowing tests to be tagged with the *known_tags*.
    """

    def tag(*tags):
        """
        Tag a test method with the given tags.
        Can be used in conjunction with the --tags command-line argument
        for runtests.py.
        """
        for t in tags:
            if t not in known_tags:
                raise ValueError('unknown tag: %r' % (t,))

        def decorate(func):
            if not callable(func) or isinstance(func, type) or (not func.__name__.startswith('test_')):
                raise TypeError('@tag(...) should be used on test methods')
            try:
                s = func.tags
            except AttributeError:
                s = func.tags = set()
            s.update(tags)
            return func
        return decorate
    return tag