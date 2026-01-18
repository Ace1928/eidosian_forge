import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_math_max_min(self):
    data_a = self._data()
    data_b = self._data(self.nlanes)
    vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
    data_max = [max(a, b) for a, b in zip(data_a, data_b)]
    simd_max = self.max(vdata_a, vdata_b)
    assert simd_max == data_max
    data_min = [min(a, b) for a, b in zip(data_a, data_b)]
    simd_min = self.min(vdata_a, vdata_b)
    assert simd_min == data_min