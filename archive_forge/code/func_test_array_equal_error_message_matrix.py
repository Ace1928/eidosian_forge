import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_array_equal_error_message_matrix():
    with pytest.raises(AssertionError) as exc_info:
        assert_equal(np.array([1, 2]), np.matrix([1, 2]))
    msg = str(exc_info.value)
    msg_reference = textwrap.dedent('\n    Arrays are not equal\n\n    (shapes (2,), (1, 2) mismatch)\n     x: array([1, 2])\n     y: matrix([[1, 2]])')
    assert_equal(msg, msg_reference)