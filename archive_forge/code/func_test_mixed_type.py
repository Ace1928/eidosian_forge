import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_mixed_type(self):
    g = r_[10.1, 1:10]
    assert_(g.dtype == 'f8')