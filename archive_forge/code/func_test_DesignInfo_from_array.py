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
def test_DesignInfo_from_array():
    di = DesignInfo.from_array([1, 2, 3])
    assert di.column_names == ['column0']
    di2 = DesignInfo.from_array([[1, 2], [2, 3], [3, 4]])
    assert di2.column_names == ['column0', 'column1']
    di3 = DesignInfo.from_array([1, 2, 3], default_column_prefix='x')
    assert di3.column_names == ['x0']
    di4 = DesignInfo.from_array([[1, 2], [2, 3], [3, 4]], default_column_prefix='x')
    assert di4.column_names == ['x0', 'x1']
    m = DesignMatrix([1, 2, 3], di3)
    assert DesignInfo.from_array(m) is di3
    m.design_info = 'asdf'
    di_weird = DesignInfo.from_array(m)
    assert di_weird.column_names == ['column0']
    import pytest
    pytest.raises(ValueError, DesignInfo.from_array, np.ones((2, 2, 2)))
    from patsy.util import have_pandas
    if have_pandas:
        import pandas
        di5 = DesignInfo.from_array(pandas.DataFrame([[1, 2]], columns=['a', 'b']))
        assert di5.column_names == ['a', 'b']
        di6 = DesignInfo.from_array(pandas.DataFrame([[1, 2]], columns=[0, 10]))
        assert di6.column_names == ['column0', 'column10']
        df = pandas.DataFrame([[1, 2]])
        df.design_info = di6
        assert DesignInfo.from_array(df) is di6