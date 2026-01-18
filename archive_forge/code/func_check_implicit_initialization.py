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
def check_implicit_initialization(self, extract_randomness):
    """
        The PRNG in new processes should be implicitly initialized
        with system entropy, to avoid reproducing the same sequences.
        """
    results = self.extract_in_processes(2, extract_randomness)
    self.check_several_outputs(results, same_expected=False)