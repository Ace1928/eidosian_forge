from __future__ import print_function
import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term
def test_SubtermInfo():
    cm = ContrastMatrix(np.ones((2, 2)), ['[1]', '[2]'])
    s = SubtermInfo(['a', 'x'], {'a': cm}, 4)
    assert s.factors == ('a', 'x')
    assert s.contrast_matrices == {'a': cm}
    assert s.num_columns == 4
    if not six.PY3:
        s = SubtermInfo(['a', 'x'], {'a': cm}, long(4))
        assert s.num_columns == 4
    repr(s)
    import pytest
    pytest.raises(TypeError, SubtermInfo, 1, {}, 1)
    pytest.raises(ValueError, SubtermInfo, ['a', 'x'], 1, 1)
    pytest.raises(ValueError, SubtermInfo, ['a', 'x'], {'z': cm}, 1)
    pytest.raises(ValueError, SubtermInfo, ['a', 'x'], {'a': 1}, 1)
    pytest.raises(ValueError, SubtermInfo, ['a', 'x'], {}, 1.5)