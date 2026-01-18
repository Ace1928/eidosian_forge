import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_confusion_map_qudits():
    q0 = cirq.LineQid(0, dimension=3)
    circuit = cirq.Circuit(cirq.XPowGate(dimension=3).on(q0) ** 1.3, cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])}), cirq.IdentityGate(qid_shape=(3,)).on(q0))
    assert_equivalent_to_deferred(circuit)