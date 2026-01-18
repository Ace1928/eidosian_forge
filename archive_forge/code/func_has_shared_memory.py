import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def has_shared_memory(self):
    """Check that created array shares data with input array."""
    if self.obj is self.arr:
        return True
    if not isinstance(self.obj, np.ndarray):
        return False
    obj_attr = wrap.array_attrs(self.obj)
    return obj_attr[0] == self.arr_attr[0]