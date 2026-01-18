from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_gate_substitution_noise_model():

    def _overrotation(op):
        if isinstance(op.gate, cirq.XPowGate):
            return cirq.XPowGate(exponent=op.gate.exponent + 0.1).on(*op.qubits)
        return op
    noise = cirq.devices.noise_model.GateSubstitutionNoiseModel(_overrotation)
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0) ** 0.5, cirq.Y(q0))
    circuit2 = cirq.Circuit(cirq.X(q0) ** 0.6, cirq.Y(q0))
    rho1 = cirq.final_density_matrix(circuit, noise=noise)
    rho2 = cirq.final_density_matrix(circuit2)
    np.testing.assert_allclose(rho1, rho2)