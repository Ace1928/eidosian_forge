from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def test_ContrastMatrix():
    cm = ContrastMatrix([[1, 0], [0, 1]], ['a', 'b'])
    assert np.array_equal(cm.matrix, np.eye(2))
    assert cm.column_suffixes == ['a', 'b']
    repr(cm)
    import pytest
    pytest.raises(PatsyError, ContrastMatrix, [[1], [0]], ['a', 'b'])
    assert_no_pickling(cm)