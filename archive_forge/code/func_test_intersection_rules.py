import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
@pytest.mark.filterwarnings('ignore:some peaks have a width of 0')
def test_intersection_rules(self):
    """Test if x == eval_height counts as an intersection."""
    x = [0, 1, 2, 1, 3, 3, 3, 1, 2, 1, 0]
    assert_allclose(peak_widths(x, peaks=[5], rel_height=0), [(0.0,), (3.0,), (5.0,), (5.0,)])
    assert_allclose(peak_widths(x, peaks=[5], rel_height=2 / 3), [(4.0,), (1.0,), (3.0,), (7.0,)])