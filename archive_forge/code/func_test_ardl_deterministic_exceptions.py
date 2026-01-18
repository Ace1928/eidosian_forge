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
def test_ardl_deterministic_exceptions(data):
    with pytest.raises(TypeError):
        ARDL(data.y, 2, data.x, 2, deterministic='seasonal')
    with pytest.warns(SpecificationWarning, match='When using deterministic, trend'):
        deterministic = DeterministicProcess(data.y.index, constant=True, order=1)
        ARDL(data.y, 2, data.x, 2, deterministic=deterministic, trend='ct')