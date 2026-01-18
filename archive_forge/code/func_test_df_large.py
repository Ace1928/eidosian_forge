import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_df_large():
    result = cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'a': np.array([[0 for _ in range(76)]] * 10000, dtype=bool), 'd': np.array([[1 for _ in range(76)]] * 10000, dtype=bool)})
    assert np.all(result.data['a'] == 0)
    assert np.all(result.data['d'] == 75557863725914323419135)
    assert result.data['a'].dtype == object
    assert result.data['d'].dtype == object