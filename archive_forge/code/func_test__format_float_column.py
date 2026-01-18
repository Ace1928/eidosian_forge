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
def test__format_float_column():

    def t(precision, numbers, expected):
        got = _format_float_column(precision, np.asarray(numbers))
        print(got, expected)
        assert np.array_equal(got, expected)
    nan_string = '%.3f' % (np.nan,)
    t(3, [1, 2.1234, 2.1239, np.nan], ['1.000', '2.123', '2.124', nan_string])
    t(3, [1, 2, 3, np.nan], ['1', '2', '3', nan_string])
    t(3, [1.0001, 2, 3, np.nan], ['1', '2', '3', nan_string])
    t(4, [1.0001, 2, 3, np.nan], ['1.0001', '2.0000', '3.0000', nan_string])