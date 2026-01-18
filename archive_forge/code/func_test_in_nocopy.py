import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
@pytest.mark.parametrize('write', ['w', 'ro'])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('inp', ['2seq', '23seq'])
def test_in_nocopy(self, write, order, inp):
    """Test if intent(in) array can be passed without copies"""
    seq = getattr(self, 'num' + inp)
    obj = np.array(seq, dtype=self.type.dtype, order=order)
    obj.setflags(write=write == 'w')
    a = self.array(obj.shape, order == 'C' and intent.in_.c or intent.in_, obj)
    assert a.has_shared_memory()