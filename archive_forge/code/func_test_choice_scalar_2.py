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
def test_choice_scalar_2(self):
    """
        Test choice(int, size)
        """
    n = 50
    pop = np.arange(n)
    self._check_choice_2(n, pop)