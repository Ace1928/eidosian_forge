import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_dephase_classical_conditions():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls('a'), cirq.measure(q1, key='b'))
    with pytest.raises(ValueError, match='defer_measurements first to remove classical controls'):
        _ = cirq.dephase_measurements(circuit)