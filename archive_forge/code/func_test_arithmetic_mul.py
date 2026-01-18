import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_arithmetic_mul(self):
    if self.sfx in ('u64', 's64'):
        return
    if self._is_fp():
        data_a = self._data()
    else:
        data_a = self._data(self._int_max() - self.nlanes)
    data_b = self._data(self._int_min(), reverse=True)
    vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
    data_mul = self.load([a * b for a, b in zip(data_a, data_b)])
    mul = self.mul(vdata_a, vdata_b)
    assert mul == data_mul