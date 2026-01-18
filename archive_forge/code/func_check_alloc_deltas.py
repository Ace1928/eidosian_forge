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
def check_alloc_deltas(deltas):
    if 3 * deltas.count(0) < len(deltas):
        return True
    if not set(deltas) <= set((1, 0, -1)):
        return True
    return False