import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize('t', np.sctypes['uint'] + np.sctypes['int'] + np.sctypes['float'])
def test_array_astype_warning(t):
    a = np.array(10, dtype=np.complex_)
    assert_warns(np.ComplexWarning, a.astype, t)