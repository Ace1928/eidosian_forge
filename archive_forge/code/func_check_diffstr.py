import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
def check_diffstr(string, pointing_at=[]):
    lines = string.splitlines()
    for item in pointing_at:
        for l in lines:
            if l.startswith('->'):
                if item in l:
                    break
        else:
            raise AssertionError('Could not find %s ' % item)