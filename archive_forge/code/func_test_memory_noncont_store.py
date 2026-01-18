import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
@pytest.mark.parametrize('intrin, elsizes, scale', [('self.storen', (32, 64), 1), ('self.storen2', (32, 64), 2)])
def test_memory_noncont_store(self, intrin, elsizes, scale):
    if self._scalar_size() not in elsizes:
        return
    npyv_storen = eval(intrin)
    data = self._data()
    vdata = self.load(data)
    hlanes = self.nlanes // scale
    for stride in range(1, 64):
        data_storen = [255] * stride * self.nlanes
        for s in range(0, hlanes * stride, stride):
            i = s // stride * scale
            data_storen[s:s + scale] = data[i:i + scale]
        storen = [255] * stride * self.nlanes
        storen += [127] * 64
        npyv_storen(storen, stride, vdata)
        assert storen[:-64] == data_storen
        assert storen[-64:] == [127] * 64
    for stride in range(-64, 0):
        data_storen = [255] * -stride * self.nlanes
        for s in range(0, hlanes * stride, stride):
            i = s // stride * scale
            data_storen[s - scale:s or None] = data[i:i + scale]
        storen = [127] * 64
        storen += [255] * -stride * self.nlanes
        npyv_storen(storen, stride, vdata)
        assert storen[64:] == data_storen
        assert storen[:64] == [127] * 64
    data_storen = [127] * self.nlanes
    storen = data_storen.copy()
    data_storen[0:scale] = data[-scale:]
    npyv_storen(storen, 0, vdata)
    assert storen == data_storen