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
@pytest.fixture(scope='module', params=[True, False])
def seasonal(request):
    return request.param