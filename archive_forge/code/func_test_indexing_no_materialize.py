from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_indexing_no_materialize(monkeypatch):

    def DecimalArray__array__(self, dtype=None):
        raise Exception('tried to convert a DecimalArray to a numpy array')
    monkeypatch.setattr(DecimalArray, '__array__', DecimalArray__array__, raising=False)
    data = make_data()
    s = pd.Series(DecimalArray(data))
    df = pd.DataFrame({'a': s, 'b': range(len(s))})
    s[s > 0.5]
    df[s > 0.5]
    s.at[0]
    df.at[0, 'a']