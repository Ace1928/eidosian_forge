import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_in_cache_from_2casttype_failure(self):
    for t in self.type.all_types():
        if t.NAME == 'STRING':
            continue
        if t.elsize >= self.type.elsize:
            continue
        obj = np.array(self.num2seq, dtype=t.dtype)
        shape = (len(self.num2seq),)
        try:
            self.array(shape, intent.in_.cache, obj)
        except ValueError as msg:
            if not str(msg).startswith('failed to initialize intent(cache) array'):
                raise
        else:
            raise SystemError('intent(cache) should have failed on smaller array')