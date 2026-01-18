import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_addition_cast_safety(self):
    """The addition method is special for the scaled float, because it
        includes the "cast" between different factors, thus cast-safety
        is influenced by the implementation.
        """
    a = self._get_array(2.0)
    b = self._get_array(-2.0)
    c = self._get_array(3.0)
    np.add(a, b, casting='equiv')
    with pytest.raises(TypeError):
        np.add(a, b, casting='no')
    with pytest.raises(TypeError):
        np.add(a, c, casting='safe')
    with pytest.raises(TypeError):
        np.add(a, a, out=c, casting='safe')