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
def normalize_vectors(num_vectors, vectors):
    normalized_vectors = np.empty((num_vectors, 2), dtype=np.float64)
    for i in range(num_vectors):
        vector = vectors[i]
        normalized_vector = normalize_2D_vector(vector)
        normalized_vectors[i, 0] = normalized_vector[0]
        normalized_vectors[i, 1] = normalized_vector[1]
    return normalized_vectors