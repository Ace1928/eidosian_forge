import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
def test_flatten_invalid_order(self):
    with pytest.raises(ValueError):
        self.conv('Z')
    for order in [False, True, 0, 8]:
        with pytest.raises(TypeError):
            self.conv(order)