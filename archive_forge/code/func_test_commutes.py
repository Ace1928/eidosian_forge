import itertools
import numpy as np
import pytest
import cirq
def test_commutes():
    for A, B in itertools.product([cirq.X, cirq.Y, cirq.Z], repeat=2):
        assert cirq.commutes(A, B) == (A == B)
    with pytest.raises(TypeError):
        assert cirq.commutes(cirq.X, 'X')
    assert cirq.commutes(cirq.X, 'X', default='default') == 'default'
    assert cirq.commutes(cirq.Z, cirq.read_json(json_text=cirq.to_json(cirq.Z)))