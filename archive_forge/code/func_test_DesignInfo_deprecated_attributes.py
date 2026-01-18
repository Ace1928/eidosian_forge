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
def test_DesignInfo_deprecated_attributes():
    d = DesignInfo(['a1', 'a2'])

    def check(attr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            assert getattr(d, attr) is d
        assert len(w) == 1
        assert w[0].category is DeprecationWarning
    check('builder')
    check('design_info')