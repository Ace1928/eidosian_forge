import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_cache_hidden(self):
    shape = (2,)
    a = self.array(shape, intent.cache.hide, None)
    assert a.arr.shape == shape
    shape = (2, 3)
    a = self.array(shape, intent.cache.hide, None)
    assert a.arr.shape == shape
    shape = (-1, 3)
    try:
        a = self.array(shape, intent.cache.hide, None)
    except ValueError as msg:
        if not str(msg).startswith('failed to create intent(cache|hide)|optional array'):
            raise
    else:
        raise SystemError('intent(cache) should have failed on undefined dimensions')