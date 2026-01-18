import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_in_copy_from_2casttype(self):
    for t in self.type.cast_types():
        obj = np.array(self.num2seq, dtype=t.dtype)
        a = self.array([len(self.num2seq)], intent.in_.copy, obj)
        assert not a.has_shared_memory()