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
def test_issue_with_refct_op_pruning(self):
    """
        GitHub Issue #1244 https://github.com/numba/numba/issues/1244
        """

    @njit
    def calculate_2D_vector_mag(vector):
        x, y = vector
        return math.sqrt(x ** 2 + y ** 2)

    @njit
    def normalize_2D_vector(vector):
        normalized_vector = np.empty(2, dtype=np.float64)
        mag = calculate_2D_vector_mag(vector)
        x, y = vector
        normalized_vector[0] = x / mag
        normalized_vector[1] = y / mag
        return normalized_vector

    @njit
    def normalize_vectors(num_vectors, vectors):
        normalized_vectors = np.empty((num_vectors, 2), dtype=np.float64)
        for i in range(num_vectors):
            vector = vectors[i]
            normalized_vector = normalize_2D_vector(vector)
            normalized_vectors[i, 0] = normalized_vector[0]
            normalized_vectors[i, 1] = normalized_vector[1]
        return normalized_vectors
    num_vectors = 10
    test_vectors = np.random.random((num_vectors, 2))
    got = normalize_vectors(num_vectors, test_vectors)
    expected = normalize_vectors.py_func(num_vectors, test_vectors)
    np.testing.assert_almost_equal(expected, got)