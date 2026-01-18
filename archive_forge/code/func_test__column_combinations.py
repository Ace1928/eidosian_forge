import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def test__column_combinations():
    assert list(_column_combinations([2, 3])) == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert list(_column_combinations([3])) == [(0,), (1,), (2,)]
    assert list(_column_combinations([])) == [()]