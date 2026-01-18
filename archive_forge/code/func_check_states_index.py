from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def check_states_index(states, ix, predicted_ix, cols):
    predicted_cov_ix = pd.MultiIndex.from_product([predicted_ix, cols]).swaplevel()
    filtered_cov_ix = pd.MultiIndex.from_product([ix, cols]).swaplevel()
    smoothed_cov_ix = pd.MultiIndex.from_product([ix, cols]).swaplevel()
    assert_(states.predicted.index.equals(predicted_ix))
    assert_(states.predicted.columns.equals(cols))
    assert_(states.predicted_cov.index.equals(predicted_cov_ix))
    assert_(states.predicted.columns.equals(cols))
    assert_(states.filtered.index.equals(ix))
    assert_(states.filtered.columns.equals(cols))
    assert_(states.filtered_cov.index.equals(filtered_cov_ix))
    assert_(states.filtered.columns.equals(cols))
    assert_(states.smoothed.index.equals(ix))
    assert_(states.smoothed.columns.equals(cols))
    assert_(states.smoothed_cov.index.equals(smoothed_cov_ix))
    assert_(states.smoothed.columns.equals(cols))