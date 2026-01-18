import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_empty_indexing():
    """Regression test for ticket 1948."""
    s = np.chararray((4,))
    assert_(s[[]].size == 0)