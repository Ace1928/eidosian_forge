from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_non_unit_range():
    idx = pd.RangeIndex(0, 700, 7)
    dp = DeterministicProcess(idx, constant=True)
    with pytest.raises(ValueError, match='The step of the index is not 1'):
        dp.range(11, 900)