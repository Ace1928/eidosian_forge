import itertools
from typing import Optional
from unittest import mock
import numpy as np
import pytest
import cirq
import cirq_google
import cirq_google.calibration.workflow as workflow
import cirq_google.calibration.xeb_wrapper
from cirq.experiments import (
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
def test_make_zeta_chi_gamma_compensation_for_moments():
    a, b = cirq.LineQubit.range(2)
    moment_allocations = [0]
    for gate_to_calibrate, engine_gate in [(cirq.FSimGate(theta=np.pi / 4, phi=0.0), SQRT_ISWAP_INV_GATE), (cirq.FSimGate(theta=-np.pi / 4, phi=0.0), SQRT_ISWAP_INV_GATE), (cirq.ISwapPowGate(exponent=0.2), cirq.FSimGate(theta=0.1 * np.pi, phi=0.0)), (cirq.PhasedFSimGate(theta=0.1, phi=0.2), cirq.FSimGate(theta=0.1, phi=0.2)), (cirq.PhasedFSimGate(theta=0.1, phi=0.2, chi=0.3), cirq.FSimGate(theta=0.1, phi=0.2)), (cirq.PhasedISwapPowGate(exponent=0.2), cirq.FSimGate(theta=0.1 * np.pi, phi=0.0)), (cirq.PhasedISwapPowGate(exponent=0.2, phase_exponent=0.4), cirq.FSimGate(theta=0.1 * np.pi, phi=0.0)), (cirq.CZ, cirq.FSimGate(theta=0.0, phi=np.pi)), (cirq.ops.CZPowGate(exponent=0.5), cirq.FSimGate(theta=0.0, phi=1.5 * np.pi)), (cirq_google.ops.SycamoreGate(), cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6))]:
        circuit = cirq.Circuit(gate_to_calibrate.on(a, b))
        characterizations = [PhasedFSimCalibrationResult(parameters={(a, b): cirq_google.PhasedFSimCharacterization(theta=engine_gate.theta, phi=engine_gate.phi, zeta=0.0, chi=0.0, gamma=0.0)}, gate=engine_gate, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)]
        calibrated_circuit = workflow.make_zeta_chi_gamma_compensation_for_moments(workflow.CircuitWithCalibration(circuit, moment_allocations), characterizations)
        assert np.allclose(cirq.unitary(circuit), cirq.unitary(calibrated_circuit.circuit))
        assert calibrated_circuit.moment_to_calibration == [None, 0, None]