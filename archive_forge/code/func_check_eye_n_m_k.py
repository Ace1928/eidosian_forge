import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def check_eye_n_m_k(self, func):
    self.check_outputs(func, [(1, 2, 0), (3, 4, 1), (3, 4, -1), (4, 3, -2), (4, 3, -5), (4, 3, 5)])