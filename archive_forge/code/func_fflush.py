import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def fflush() -> None:
    """
    C code in some solvers uses libc buffering; if we want to capture log output from
    those solvers to use in tests, we must flush the libc buffers before trying to read
    the log contents from python.
    https://github.com/pytest-dev/pytest/issues/8753
    """
    import ctypes
    libc = ctypes.CDLL(None)
    libc.fflush(None)