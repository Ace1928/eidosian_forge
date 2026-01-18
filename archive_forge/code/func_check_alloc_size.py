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
def check_alloc_size(self, pyfunc):
    """Checks that pyfunc will error, not segfaulting due to array size."""
    cfunc = nrtjit(pyfunc)
    with self.assertRaises(ValueError) as e:
        cfunc()
    self.assertIn('array is too big', str(e.exception))