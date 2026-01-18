import numpy as np
import cirq
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
def get_expected_cm(num_qubits: int, p0: float, p1: float):
    expected_cm = np.zeros((2 ** num_qubits,) * 2)
    for i in range(2 ** num_qubits):
        for j in range(2 ** num_qubits):
            p = 1.0
            for k in range(num_qubits):
                b0 = i >> k & 1
                b1 = j >> k & 1
                if b0 == 0:
                    p *= p0 * b1 + (1 - p0) * (1 - b1)
                else:
                    p *= p1 * (1 - b1) + (1 - p1) * b1
            expected_cm[i][j] = p
    return expected_cm