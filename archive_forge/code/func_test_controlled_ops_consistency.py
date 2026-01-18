import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('gate', [cirq.CCX, cirq.CCZ, cirq.CSWAP])
def test_controlled_ops_consistency(gate):
    a, b, c, d = cirq.LineQubit.range(4)
    assert gate.controlled(0) is gate
    assert gate(a, b, c).controlled_by(d) == gate(d, b, c).controlled_by(a)