import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__
def test_conversion_boolean(self):
    bsfx = 'b' + self.sfx[1:]
    to_boolean = getattr(self.npyv, 'cvt_%s_%s' % (bsfx, self.sfx))
    from_boolean = getattr(self.npyv, 'cvt_%s_%s' % (self.sfx, bsfx))
    false_vb = to_boolean(self.setall(0))
    true_vb = self.cmpeq(self.setall(0), self.setall(0))
    assert false_vb != true_vb
    false_vsfx = from_boolean(false_vb)
    true_vsfx = from_boolean(true_vb)
    assert false_vsfx != true_vsfx