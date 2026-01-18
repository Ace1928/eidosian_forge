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
def fixup_case(self, case):
    """
        Remove any unpicklable attributes from TestCase instance *case*.
        """
    case._outcomeForDoCleanups = None