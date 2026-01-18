from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
def check_impact_indices(news, impact_dates, impacted_variables):
    for attr in ['total_impacts', 'update_impacts', 'revision_impacts', 'post_impacted_forecasts', 'prev_impacted_forecasts']:
        val = getattr(news, attr)
        assert_(val.index.equals(impact_dates))
        assert_equal(val.columns.tolist(), impacted_variables)