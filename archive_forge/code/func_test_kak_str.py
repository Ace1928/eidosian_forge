import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_kak_str():
    v = cirq.KakDecomposition(interaction_coefficients=(0.3 * np.pi / 4, 0.2 * np.pi / 4, 0.1 * np.pi / 4), single_qubit_operations_before=(cirq.unitary(cirq.I), cirq.unitary(cirq.X)), single_qubit_operations_after=(cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)), global_phase=1j)
    assert str(v) == 'KAK {\n    xyz*(4/π): 0.3, 0.2, 0.1\n    before: (0*π around X) ⊗ (1*π around X)\n    after: (1*π around Y) ⊗ (1*π around Z)\n}'