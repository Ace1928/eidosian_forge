import os
import sys
from packaging.version import Version, parse
import numpy as np
from numpy.testing import assert_allclose, assert_
import pandas as pd
def check_fitted(results):
    import pytest
    from statsmodels.genmod.generalized_linear_model import GLMResults
    from statsmodels.discrete.discrete_model import DiscreteResults
    results = getattr(results, '_results', results)
    if isinstance(results, (GLMResults, DiscreteResults)):
        pytest.skip(f'Not supported for {type(results)}')
    res = results
    fitted = res.fittedvalues
    assert_allclose(res.model.endog - fitted, res.resid, rtol=1e-12)
    assert_allclose(fitted, res.predict(), rtol=1e-12)