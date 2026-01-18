import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
@jit(nopython=True, nogil=True)
def py_extract_randomness(seed, out):
    if seed != 0:
        random.seed(seed)
    for i in range(out.size):
        out[i] = random.getrandbits(32)