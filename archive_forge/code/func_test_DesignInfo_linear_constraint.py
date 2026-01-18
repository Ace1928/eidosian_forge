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
def test_DesignInfo_linear_constraint():
    di = DesignInfo(['a1', 'a2', 'a3', 'b'])
    con = di.linear_constraint(['2 * a1 = b + 1', 'a3'])
    assert con.variable_names == ['a1', 'a2', 'a3', 'b']
    assert np.all(con.coefs == [[2, 0, 0, -1], [0, 0, 1, 0]])
    assert np.all(con.constants == [[1], [0]])