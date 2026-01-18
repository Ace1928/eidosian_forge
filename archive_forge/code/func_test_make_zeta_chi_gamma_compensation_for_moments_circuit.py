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
def test_make_zeta_chi_gamma_compensation_for_moments_circuit():
    a, b = cirq.LineQubit.range(2)
    characterizations = [PhasedFSimCalibrationResult(parameters={(a, b): SQRT_ISWAP_INV_PARAMETERS}, gate=SQRT_ISWAP_INV_GATE, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)]
    for circuit, expected_moment_to_calibration in [(cirq.Circuit(cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b)), [None, 0, None]), (cirq.Circuit([cirq.Z.on(a), cirq.FSimGate(theta=-np.pi / 4, phi=0.0).on(a, b)]), [None, None, 0, None])]:
        calibrated_circuit = workflow.make_zeta_chi_gamma_compensation_for_moments(circuit, characterizations)
        assert np.allclose(cirq.unitary(circuit), cirq.unitary(calibrated_circuit.circuit))
        assert calibrated_circuit.moment_to_calibration == expected_moment_to_calibration