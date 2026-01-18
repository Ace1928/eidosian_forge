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
@pytest.mark.matplotlib
def test_diagnostics_plot(data, close_figures):
    import matplotlib.figure
    res = ARDL(data.y, 2, data.x, {'lry': 3, 'ibo': 2, 'ide': [1, 3]}, trend='ct', seasonal=True).fit()
    fig = res.plot_diagnostics()
    assert isinstance(fig, matplotlib.figure.Figure)