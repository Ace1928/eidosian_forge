import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_multi_measurement_histogram():
    result = cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=bool), 'c': np.array([[0], [0], [1], [0], [1]], dtype=bool)})
    assert result.multi_measurement_histogram(keys=['ab']) == collections.Counter({(1,): 4, (2,): 1})
    assert result.multi_measurement_histogram(keys=['c']) == collections.Counter({(0,): 3, (1,): 2})
    assert result.multi_measurement_histogram(keys=['ab', 'c']) == collections.Counter({(1, 0): 2, (1, 1): 2, (2, 0): 1})
    assert result.multi_measurement_histogram(keys=[], fold_func=lambda e: None) == collections.Counter({None: 5})
    assert result.multi_measurement_histogram(keys=['ab'], fold_func=lambda e: None) == collections.Counter({None: 5})
    assert result.multi_measurement_histogram(keys=['ab', 'c'], fold_func=lambda e: None) == collections.Counter({None: 5})
    assert result.multi_measurement_histogram(keys=['ab', 'c'], fold_func=lambda e: tuple((tuple(f) for f in e))) == collections.Counter({((False, True), (False,)): 2, ((False, True), (True,)): 2, ((True, False), (False,)): 1})