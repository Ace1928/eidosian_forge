from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
def index_range(*args, **kwargs):
    return pd.period_range(*args, **kwargs).to_timestamp(freq='Q')