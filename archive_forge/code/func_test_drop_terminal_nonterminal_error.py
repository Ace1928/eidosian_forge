import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_drop_terminal_nonterminal_error():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q0, q1, key='a~b', invert_mask=[0, 1]), cirq.CX(q0, q1))))
    with pytest.raises(ValueError, match='Circuit contains a non-terminal measurement'):
        _ = cirq.drop_terminal_measurements(circuit)
    with pytest.raises(ValueError, match='Context has `deep=False`'):
        _ = cirq.drop_terminal_measurements(circuit, context=cirq.TransformerContext(deep=False))
    with pytest.raises(ValueError, match='Context has `deep=False`'):
        _ = cirq.drop_terminal_measurements(circuit, context=None)