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
def extract_in_processes(self, nprocs, extract_randomness):
    """
        Run *nprocs* processes extracting randomness
        without explicit seeding.
        """
    q = multiprocessing.Queue()
    results = []

    def target_inner():
        out = self._get_output(self._extract_iterations)
        extract_randomness(seed=0, out=out)
        return out

    def target():
        try:
            out = target_inner()
            q.put(out)
        except Exception as e:
            q.put(e)
            raise
    if hasattr(multiprocessing, 'get_context'):
        mpc = multiprocessing.get_context('fork')
    else:
        mpc = multiprocessing
    procs = [mpc.Process(target=target) for i in range(nprocs)]
    for p in procs:
        p.start()
    for i in range(nprocs):
        results.append(q.get(timeout=5))
    for p in procs:
        p.join()
    results.append(target_inner())
    for res in results:
        if isinstance(res, Exception):
            self.fail('Exception in child: %s' % (res,))
    return results