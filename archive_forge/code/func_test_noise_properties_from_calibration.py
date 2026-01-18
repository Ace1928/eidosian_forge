import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier
from google.protobuf.text_format import Merge
import numpy as np
import pytest
def test_noise_properties_from_calibration():
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    pauli_error = [0.001, 0.002, 0.003]
    incoherent_error = [0.0001, 0.0002, 0.0003]
    p00_error = [0.004, 0.005, 0.006]
    p11_error = [0.007, 0.008, 0.009]
    t1_micros = [10, 20, 30]
    syc_pauli = [0.01, 0.02]
    iswap_pauli = [0.03, 0.04]
    syc_angles = [cirq.PhasedFSimGate(theta=0.011, phi=-0.021), cirq.PhasedFSimGate(theta=-0.012, phi=0.022)]
    iswap_angles = [cirq.PhasedFSimGate(theta=-0.013, phi=0.023), cirq.PhasedFSimGate(theta=0.014, phi=-0.024)]
    calibration = get_mock_calibration(pauli_error, incoherent_error, p00_error, p11_error, t1_micros, syc_pauli, iswap_pauli, syc_angles, iswap_angles)
    prop = cirq_google.noise_properties_from_calibration(calibration)
    for i, q in enumerate(qubits):
        assert np.isclose(prop.gate_pauli_errors[OpIdentifier(cirq.PhasedXZGate, q)], pauli_error[i])
        assert np.allclose(prop.readout_errors[q], np.array([p00_error[i], p11_error[i]]))
        assert np.isclose(prop.t1_ns[q], t1_micros[i] * 1000)
        microwave_time_ns = 25.0
        tphi_err = incoherent_error[i] - microwave_time_ns / (3 * prop.t1_ns[q])
        if tphi_err > 0:
            tphi_ns = microwave_time_ns / (3 * tphi_err)
        else:
            tphi_ns = 10000000000.0
        assert prop.tphi_ns[q] == tphi_ns
    qubit_pairs = [(qubits[0], qubits[1]), (qubits[0], qubits[2])]
    for i, qs in enumerate(qubit_pairs):
        for gate, values in [(cirq_google.SycamoreGate, syc_pauli), (cirq.ISwapPowGate, iswap_pauli)]:
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs)], values[i])
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs[::-1])], values[i])
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs)], values[i])
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs[::-1])], values[i])
        for gate, values in [(cirq_google.SycamoreGate, syc_angles), (cirq.ISwapPowGate, iswap_angles)]:
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]