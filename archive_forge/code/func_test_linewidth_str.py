import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_linewidth_str(self):
    a = np.full(18, fill_value=2)
    np.set_printoptions(linewidth=18)
    assert_equal(str(a), textwrap.dedent('            [2 2 2 2 2 2 2 2\n             2 2 2 2 2 2 2 2\n             2 2]'))
    np.set_printoptions(linewidth=18, legacy='1.13')
    assert_equal(str(a), textwrap.dedent('            [2 2 2 2 2 2 2 2 2\n             2 2 2 2 2 2 2 2 2]'))