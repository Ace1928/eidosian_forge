import os
import sys
import copy
import platform
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal
from numpy.core.multiarray import typeinfo as _typeinfo
from . import util
def test_c_in_from_23seq(self):
    a = self.array([len(self.num23seq), len(self.num23seq[0])], intent.in_, self.num23seq)
    assert not a.has_shared_memory()