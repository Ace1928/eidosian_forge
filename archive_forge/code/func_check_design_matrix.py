from __future__ import print_function
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import (atleast_2d_column_default,
from patsy.desc import Term, INTERCEPT
from patsy.build import *
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo
def check_design_matrix(mm, expected_rank, termlist, column_names=None):
    assert_full_rank(mm)
    assert set(mm.design_info.terms) == set(termlist)
    if column_names is not None:
        assert mm.design_info.column_names == column_names
    assert mm.ndim == 2
    assert mm.shape[1] == expected_rank