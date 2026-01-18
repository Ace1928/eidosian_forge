from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_no_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    del class_kwargs['period']
    class_kwargs['endog'] = pd.Series(class_kwargs['endog'])
    with pytest.raises(ValueError, match='Unable to determine period from'):
        STL(**class_kwargs)