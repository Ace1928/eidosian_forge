import pytest
import numpy as np
from numpy.testing import assert_, assert_raises
def test_gh_15395(self):
    x = B1(1.0)
    assert_(str(x) == '1.0')
    with pytest.raises(TypeError):
        B1(1.0, 2.0)