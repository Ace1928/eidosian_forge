import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
@njit
def normalize_2D_vector(vector):
    normalized_vector = np.empty(2, dtype=np.float64)
    mag = calculate_2D_vector_mag(vector)
    x, y = vector
    normalized_vector[0] = x / mag
    normalized_vector[1] = y / mag
    return normalized_vector