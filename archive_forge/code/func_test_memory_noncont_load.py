import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin, elsizes, scale', [('self.loadn', (32, 64), 1), ('self.loadn2', (32, 64), 2)])
def test_memory_noncont_load(self, intrin, elsizes, scale):
    if self._scalar_size() not in elsizes:
        return
    npyv_loadn = eval(intrin)
    for stride in range(-64, 64):
        if stride < 0:
            data = self._data(stride, -stride * self.nlanes)
            data_stride = list(itertools.chain(*zip(*[data[-i::stride] for i in range(scale, 0, -1)])))
        elif stride == 0:
            data = self._data()
            data_stride = data[0:scale] * (self.nlanes // scale)
        else:
            data = self._data(count=stride * self.nlanes)
            data_stride = list(itertools.chain(*zip(*[data[i::stride] for i in range(scale)])))
        data_stride = self.load(data_stride)
        loadn = npyv_loadn(data, stride)
        assert loadn == data_stride