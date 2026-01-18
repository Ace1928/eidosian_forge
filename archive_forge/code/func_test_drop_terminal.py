import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_drop_terminal():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CX(q0, q1), cirq.measure(q0, q1, key='a~b', invert_mask=[0, 1]))))
    dropped = cirq.drop_terminal_measurements(circuit)
    cirq.testing.assert_same_circuits(dropped, cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CX(q0, q1), cirq.I(q0), cirq.X(q1)))))