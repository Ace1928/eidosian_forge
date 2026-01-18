from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
def estimate_count(Ms, b):
    m = 1 << b
    M = reduce_state(Ms, b)
    alpha = 0.7213 / (1 + 1.079 / m)
    E = alpha * m / (2.0 ** (-M.astype('f8'))).sum() * m
    if E < 2.5 * m:
        V = (M == 0).sum()
        if V:
            return m * np.log(m / V)
    if E > 2 ** 32 / 30.0:
        return -2 ** 32 * np.log1p(-E / 2 ** 32)
    return E