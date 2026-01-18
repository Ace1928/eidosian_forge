import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_conditional_mnlogit_2d():
    df = gen_mnlogit(90)
    model = ConditionalMNLogit.from_formula('y ~ 0 + x1 + x2', groups='g', data=df)
    result = model.fit()
    assert_allclose(result.params, np.asarray([[0.75592035, -1.58565494], [1.82919869, -1.32594231]]), rtol=1e-05, atol=1e-05)
    assert_allclose(result.bse, np.asarray([[0.68099698, 0.70142727], [0.65190315, 0.59653771]]), rtol=1e-05, atol=1e-05)