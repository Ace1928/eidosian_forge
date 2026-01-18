import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin, elsizes, scale, fill', [('self.load_tillz, self.load_till', (32, 64), 1, [65535]), ('self.load2_tillz, self.load2_till', (32, 64), 2, [65535, 32767])])
def test_memory_partial_load(self, intrin, elsizes, scale, fill):
    if self._scalar_size() not in elsizes:
        return
    npyv_load_tillz, npyv_load_till = eval(intrin)
    data = self._data()
    lanes = list(range(1, self.nlanes + 1))
    lanes += [self.nlanes ** 2, self.nlanes ** 4]
    for n in lanes:
        load_till = npyv_load_till(data, n, *fill)
        load_tillz = npyv_load_tillz(data, n)
        n *= scale
        data_till = data[:n] + fill * ((self.nlanes - n) // scale)
        assert load_till == data_till
        data_tillz = data[:n] + [0] * (self.nlanes - n)
        assert load_tillz == data_tillz