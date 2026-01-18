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
def check_several_outputs(self, results, same_expected):
    for out in results:
        self.check_output(out)
    if same_expected:
        expected_distinct = 1
    else:
        expected_distinct = len(results)
    heads = {tuple(out[:5]) for out in results}
    tails = {tuple(out[-5:]) for out in results}
    sums = {out.sum() for out in results}
    self.assertEqual(len(heads), expected_distinct, heads)
    self.assertEqual(len(tails), expected_distinct, tails)
    self.assertEqual(len(sums), expected_distinct, sums)