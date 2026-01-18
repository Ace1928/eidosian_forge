import sys
import __future__
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc, Term, INTERCEPT
from patsy.categorical import C
from patsy.contrasts import Helmert
from patsy.user_util import balanced, LookupFactor
from patsy.build import (design_matrix_builders,
from patsy.highlevel import *
from patsy.util import (have_pandas,
from patsy.origin import Origin
def test_env_not_saved_in_builder():
    x_in_env = [1, 2, 3]
    design_matrix = dmatrix('x_in_env', {})
    x_in_env = [10, 20, 30]
    design_matrix2 = dmatrix(design_matrix.design_info, {})
    assert np.allclose(design_matrix, design_matrix2)