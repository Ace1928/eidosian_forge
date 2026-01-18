import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def assert_path_equal(self, comp, benchmark):
    ret = len(comp) == len(benchmark)
    assert_(ret)
    for pos in range(len(comp) - 1):
        ret &= isinstance(comp[pos + 1], tuple)
        ret &= comp[pos + 1] == benchmark[pos + 1]
    assert_(ret)