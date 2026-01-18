import numpy as np
import pytest
from pandas import (
@pytest.fixture
def series_mixed():
    return Series(np.random.default_rng(2).standard_normal(4), index=[2, 4, 'null', 8])