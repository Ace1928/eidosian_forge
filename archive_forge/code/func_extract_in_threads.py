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
def extract_in_threads(self, nthreads, extract_randomness, seed):
    """
        Run *nthreads* threads extracting randomness with the given *seed*
        (no seeding if 0).
        """
    results = [self._get_output(self._extract_iterations) for i in range(nthreads + 1)]

    def target(i):
        extract_randomness(seed=seed, out=results[i])
    threads = [threading.Thread(target=target, args=(i,)) for i in range(nthreads)]
    for th in threads:
        th.start()
    target(nthreads)
    for th in threads:
        th.join()
    return results