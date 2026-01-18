import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_from_text():
    d = cirq.DensePauliString
    m = cirq.MutableDensePauliString
    assert d('') == d(pauli_mask=[])
    assert m('') == m(pauli_mask=[])
    assert d('YYXYY') == d([2, 2, 1, 2, 2])
    assert d('XYZI') == d([1, 2, 3, 0])
    assert d('III', coefficient=-1) == d([0, 0, 0], coefficient=-1)
    assert d('XXY', coefficient=1j) == d([1, 1, 2], coefficient=1j)
    assert d('ixyz') == d([0, 1, 2, 3])
    assert d(['i', 'x', 'y', 'z']) == d([0, 1, 2, 3])
    with pytest.raises(TypeError, match='Expected a cirq.PAULI_GATE_LIKE'):
        _ = d('2')