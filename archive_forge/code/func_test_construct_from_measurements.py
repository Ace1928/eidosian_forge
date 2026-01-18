import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_construct_from_measurements():
    r = cirq.ResultDict(params=None, measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])})
    assert np.all(r.measurements['a'] == np.array([[0, 0], [1, 1]]))
    assert np.all(r.measurements['b'] == np.array([[0, 0, 0], [1, 1, 1]]))
    assert np.all(r.records['a'] == np.array([[[0, 0]], [[1, 1]]]))
    assert np.all(r.records['b'] == np.array([[[0, 0, 0]], [[1, 1, 1]]]))