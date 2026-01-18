import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_key_unset(sim):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls('a'), cirq.measure(q1, key='b'))
    result = sim.run(circuit)
    assert result.measurements['a'] == 0
    assert result.measurements['b'] == 0