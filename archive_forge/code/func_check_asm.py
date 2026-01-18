import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
def check_asm(self, pyfunc, *args, **kwargs):
    std_pattern = kwargs.pop('std_pattern', None)
    fast_pattern = kwargs.pop('fast_pattern', None)
    jitstd, jitfast = self.compile(pyfunc, *args)
    if std_pattern:
        self.check_svml_presence(jitstd, std_pattern)
    if fast_pattern:
        self.check_svml_presence(jitfast, fast_pattern)