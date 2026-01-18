import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_imul():
    f = cirq.DensePauliString
    m = cirq.MutableDensePauliString
    p = f('III')
    p2 = p
    p2 *= 2
    assert p.coefficient == 1
    assert p is not p2
    p = m('III')
    p2 = p
    p2 *= 2
    assert p.coefficient == 2
    assert p is p2
    p *= f('X')
    assert p == m('XII', coefficient=2)
    p *= m('XY')
    assert p == m('IYI', coefficient=2)
    p *= 1j
    assert p == m('IYI', coefficient=2j)
    p *= 0.5
    assert p == m('IYI', coefficient=1j)
    p *= cirq.X(cirq.LineQubit(1))
    assert p == m('IZI')
    with pytest.raises(ValueError, match='smaller than'):
        p *= f('XXXXXXXXXXXX')
    with pytest.raises(TypeError):
        p *= object()

    class UnknownNumber(numbers.Number):
        pass
    with pytest.raises(TypeError):
        p *= UnknownNumber()