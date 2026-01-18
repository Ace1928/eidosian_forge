import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_f_inout_23seq(self):
    obj = np.array(self.num23seq, dtype=self.type.dtype, order='F')
    shape = (len(self.num23seq), len(self.num23seq[0]))
    a = self.array(shape, intent.in_.inout, obj)
    assert a.has_shared_memory()
    obj = np.array(self.num23seq, dtype=self.type.dtype, order='C')
    shape = (len(self.num23seq), len(self.num23seq[0]))
    try:
        a = self.array(shape, intent.in_.inout, obj)
    except ValueError as msg:
        if not str(msg).startswith('failed to initialize intent(inout) array'):
            raise
    else:
        raise SystemError('intent(inout) should have failed on improper array')