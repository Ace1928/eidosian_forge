import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_in_cache_from_2casttype(self):
    for t in self.type.all_types():
        if t.elsize != self.type.elsize:
            continue
        obj = np.array(self.num2seq, dtype=t.dtype)
        shape = (len(self.num2seq),)
        a = self.array(shape, intent.in_.c.cache, obj)
        assert a.has_shared_memory()
        a = self.array(shape, intent.in_.cache, obj)
        assert a.has_shared_memory()
        obj = np.array(self.num2seq, dtype=t.dtype, order='F')
        a = self.array(shape, intent.in_.c.cache, obj)
        assert a.has_shared_memory()
        a = self.array(shape, intent.in_.cache, obj)
        assert a.has_shared_memory(), repr(t.dtype)
        try:
            a = self.array(shape, intent.in_.cache, obj[::-1])
        except ValueError as msg:
            if not str(msg).startswith('failed to initialize intent(cache) array'):
                raise
        else:
            raise SystemError('intent(cache) should have failed on multisegmented array')