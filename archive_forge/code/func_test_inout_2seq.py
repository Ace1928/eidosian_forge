import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_inout_2seq(self):
    obj = np.array(self.num2seq, dtype=self.type.dtype)
    a = self.array([len(self.num2seq)], intent.inout, obj)
    assert a.has_shared_memory()
    try:
        a = self.array([2], intent.in_.inout, self.num2seq)
    except TypeError as msg:
        if not str(msg).startswith('failed to initialize intent(inout|inplace|cache) array'):
            raise
    else:
        raise SystemError('intent(inout) should have failed on sequence')