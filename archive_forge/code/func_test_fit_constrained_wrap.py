from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import (
def test_fit_constrained_wrap(self):
    res2 = self.res2
    from statsmodels.base._constraints import fit_constrained_wrap
    res_wrap = fit_constrained_wrap(self.res1m.model, self.constraints_rq)
    assert_allclose(res_wrap.params, res2.params, rtol=1e-06)
    assert_allclose(res_wrap.params, res2.params, rtol=1e-06)