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
def test_make_zeta_chi_gamma_compensation_for_moments_wrong_engine_gate_error():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.FSimGate(theta=np.pi / 4, phi=0.2).on(a, b))
    characterizations = [PhasedFSimCalibrationResult(parameters={(a, b): cirq_google.PhasedFSimCharacterization(theta=np.pi / 4, phi=0.2, zeta=0.0, chi=0.0, gamma=0.0)}, gate=SQRT_ISWAP_INV_GATE, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)]
    with pytest.raises(ValueError, match="Engine gate .+ doesn't match characterized gate .+"):
        workflow.make_zeta_chi_gamma_compensation_for_moments(workflow.CircuitWithCalibration(circuit, [0]), characterizations)