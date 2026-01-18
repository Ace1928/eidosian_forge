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
def test_eval_env_type_builder():
    data = {'x': [1, 2, 3]}

    def iter_maker():
        yield data
    pytest.raises(TypeError, design_matrix_builders, [make_termlist('x')], iter_maker, 'foo')