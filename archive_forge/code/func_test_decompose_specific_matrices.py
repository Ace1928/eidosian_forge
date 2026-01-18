import numpy as np
import scipy.stats
import cirq
def test_decompose_specific_matrices():
    for gate in [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.I, cirq.T, cirq.S]:
        for controls_count in range(7):
            _test_decompose(cirq.unitary(gate), controls_count)