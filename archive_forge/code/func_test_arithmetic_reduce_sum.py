import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_arithmetic_reduce_sum(self):
    """
        Test reduce sum intrinsics:
            npyv_sum_##sfx
        """
    if self.sfx not in ('u32', 'u64', 'f32', 'f64'):
        return
    data = self._data()
    vdata = self.load(data)
    data_sum = sum(data)
    vsum = self.sum(vdata)
    assert vsum == data_sum