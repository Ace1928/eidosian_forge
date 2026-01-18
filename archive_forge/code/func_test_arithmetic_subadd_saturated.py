import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_arithmetic_subadd_saturated(self):
    if self.sfx in ('u32', 's32', 'u64', 's64'):
        return
    data_a = self._data(self._int_max() - self.nlanes)
    data_b = self._data(self._int_min(), reverse=True)
    vdata_a, vdata_b = (self.load(data_a), self.load(data_b))
    data_adds = self._int_clip([a + b for a, b in zip(data_a, data_b)])
    adds = self.adds(vdata_a, vdata_b)
    assert adds == data_adds
    data_subs = self._int_clip([a - b for a, b in zip(data_a, data_b)])
    subs = self.subs(vdata_a, vdata_b)
    assert subs == data_subs