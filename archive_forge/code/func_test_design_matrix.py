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
def test_design_matrix():
    import pytest
    di = DesignInfo(['a1', 'a2', 'a3', 'b'])
    mm = DesignMatrix([[12, 14, 16, 18]], di)
    assert mm.design_info.column_names == ['a1', 'a2', 'a3', 'b']
    bad_di = DesignInfo(['a1'])
    pytest.raises(ValueError, DesignMatrix, [[12, 14, 16, 18]], bad_di)
    mm2 = DesignMatrix([[12, 14, 16, 18]])
    assert mm2.design_info.column_names == ['column0', 'column1', 'column2', 'column3']
    mm3 = DesignMatrix([12, 14, 16, 18])
    assert mm3.shape == (4, 1)
    pytest.raises(ValueError, DesignMatrix, [[[1]]])
    mm4 = DesignMatrix(mm)
    assert mm4 is mm
    mm5 = DesignMatrix(mm.diagonal())
    assert mm5 is not mm
    mm6 = DesignMatrix([[12, 14, 16, 18]], default_column_prefix='x')
    assert mm6.design_info.column_names == ['x0', 'x1', 'x2', 'x3']
    assert_no_pickling(mm6)
    pytest.raises(ValueError, DesignMatrix, [1, 2, 3j])
    pytest.raises(ValueError, DesignMatrix, ['a', 'b', 'c'])
    pytest.raises(ValueError, DesignMatrix, [1, 2, object()])
    repr(mm)
    repr(DesignMatrix(np.arange(100)))
    repr(DesignMatrix(np.arange(100) * 2.0))
    repr(mm[1:, :])
    repr(DesignMatrix(np.arange(100).reshape((1, 100))))
    repr(DesignMatrix([np.nan, np.inf]))
    repr(DesignMatrix([np.nan, 0, 1e+20, 20.5]))
    repr(DesignMatrix(np.zeros((1, 0))))
    repr(DesignMatrix(np.zeros((0, 1))))
    repr(DesignMatrix(np.zeros((0, 0))))