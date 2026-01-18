import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_key_set_in_subcircuit_outer_scope():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='a'), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q1).with_classical_controls('a'))))
    circuit.append(cirq.measure(q1, key='b'))
    result = cirq.Simulator().run(circuit)
    assert result.measurements['a'] == 1
    assert result.measurements['b'] == 1