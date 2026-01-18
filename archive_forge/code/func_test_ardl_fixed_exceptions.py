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
def test_ardl_fixed_exceptions(data):
    fixed = np.random.standard_normal((2, 200))
    with pytest.raises(ValueError, match='fixed must be an'):
        ARDL(data.y, 2, data.x, 2, fixed=fixed)
    fixed = np.random.standard_normal((dane_data.lrm.shape[0], 2))
    fixed[20, 0] = -np.inf
    with pytest.raises(ValueError, match='fixed must be an'):
        ARDL(data.y, 2, data.x, 2, fixed=fixed)