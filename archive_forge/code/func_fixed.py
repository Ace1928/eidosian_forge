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
@pytest.fixture(scope='module', params=[None, 2])
def fixed(request):
    if request.param is None:
        return None
    index = dane_data.lrm.index
    gen = np.random.default_rng(0)
    return pd.DataFrame(gen.standard_t(10, (dane_data.lrm.shape[0], 2)), index=index, columns=['z0', 'z1'])