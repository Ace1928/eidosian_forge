import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_dephase_nocompile_context():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CX(q1, q0), cirq.measure(q0, key='a').with_tags('nocompile'), cirq.CX(q0, q1), cirq.measure(q1, key='b'))))
    dephased = cirq.dephase_measurements(circuit, context=cirq.TransformerContext(deep=True, tags_to_ignore=('nocompile',)))
    cirq.testing.assert_same_circuits(dephased, cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CX(q1, q0), cirq.measure(q0, key='a').with_tags('nocompile'), cirq.CX(q0, q1), cirq.KrausChannel.from_channel(cirq.phase_damp(1), key='b')(q1)))))