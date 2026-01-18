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
def t_incremental(data1, data2):

    def iter_maker():
        yield {'x': data1}
        yield {'x': data2}
    try:
        builders = design_matrix_builders([termlist], iter_maker, 0)
        build_design_matrices(builders, {'x': data1})
        build_design_matrices(builders, {'x': data2})
    except PatsyError:
        pass
    else:
        raise AssertionError