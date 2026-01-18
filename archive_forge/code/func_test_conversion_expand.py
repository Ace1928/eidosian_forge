import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_conversion_expand(self):
    """
        Test expand intrinsics:
            npyv_expand_u16_u8
            npyv_expand_u32_u16
        """
    if self.sfx not in ('u8', 'u16'):
        return
    totype = self.sfx[0] + str(int(self.sfx[1:]) * 2)
    expand = getattr(self.npyv, f'expand_{totype}_{self.sfx}')
    data = self._data(self._int_max() - self.nlanes)
    vdata = self.load(data)
    edata = expand(vdata)
    data_lo = data[:self.nlanes // 2]
    data_hi = data[self.nlanes // 2:]
    assert edata == (data_lo, data_hi)