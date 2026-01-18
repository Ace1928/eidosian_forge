from typing import NamedTuple
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest
from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl.model import (
from statsmodels.tsa.deterministic import DeterministicProcess
@pytest.mark.parametrize('case', [1, 2, 3, 4, 5])
def test_bounds_test(case):
    mod = UECM(dane_data.lrm, 3, dane_data[['lry', 'ibo', 'ide']], {'lry': 1, 'ibo': 3, 'ide': 2})
    res = mod.fit()
    expected = {1: 0.7109023, 2: 5.116768, 3: 6.205875, 4: 5.430622, 5: 6.785325}
    bounds_result = res.bounds_test(case)
    assert_allclose(bounds_result.stat, expected[case])
    assert 'BoundsTestResult' in str(bounds_result)