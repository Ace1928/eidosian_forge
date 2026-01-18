from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_fourier(index):
    f = Fourier(period=12, order=3)
    terms = f.in_sample(index)
    assert f.order == 3
    assert terms.shape == (index.shape[0], 2 * f.order)
    loc = np.arange(index.shape[0]) / 12
    for i, col in enumerate(terms):
        j = i // 2 + 1
        fn = np.cos if i % 2 else np.sin
        expected = fn(2 * np.pi * j * loc)
        np.testing.assert_allclose(terms[col], expected, atol=1e-08)
    cols = []
    for i in range(2 * f.order):
        fn = 'cos' if i % 2 else 'sin'
        cols.append(f'{fn}({i // 2 + 1},12)')
    assert list(terms.columns) == cols