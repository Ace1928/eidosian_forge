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
@pytest.mark.parametrize('theta,zeta,chi,gamma,phi', itertools.product([0.1, 0.7], [-0.3, 0.1, 0.5], [-0.3, 0.2, 0.4], [-0.6, 0.1, 0.6], [0.2, 0.6]))
def test_fsim_phase_corrections(theta: float, zeta: float, chi: float, gamma: float, phi: float) -> None:
    a, b = cirq.LineQubit.range(2)
    expected_gate = cirq.PhasedFSimGate(theta=theta, zeta=-zeta, chi=-chi, gamma=-gamma, phi=phi)
    expected = cirq.unitary(expected_gate)
    corrected = workflow.FSimPhaseCorrections.from_characterization((a, b), PhaseCalibratedFSimGate(cirq.FSimGate(theta=theta, phi=phi), 0.0), cirq_google.PhasedFSimCharacterization(theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi), characterization_index=5)
    actual = cirq.unitary(corrected.as_circuit())
    assert cirq.equal_up_to_global_phase(actual, expected)
    assert corrected.moment_to_calibration == [None, 5, None]