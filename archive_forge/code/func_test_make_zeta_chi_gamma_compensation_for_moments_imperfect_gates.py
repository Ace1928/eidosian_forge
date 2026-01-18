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
def test_make_zeta_chi_gamma_compensation_for_moments_imperfect_gates():
    params_cz_ab = cirq_google.PhasedFSimCharacterization(zeta=0.02, chi=0.05, gamma=0.04, theta=0.0, phi=np.pi)
    params_cz_cd = cirq_google.PhasedFSimCharacterization(zeta=0.03, chi=0.08, gamma=0.03, theta=0.0, phi=np.pi)
    params_syc_ab = cirq_google.PhasedFSimCharacterization(zeta=0.01, chi=0.09, gamma=0.02, theta=np.pi / 2, phi=np.pi / 6)
    params_sqrt_iswap_ac = cirq_google.PhasedFSimCharacterization(zeta=0.05, chi=0.06, gamma=0.07, theta=np.pi / 4, phi=0.0)
    params_sqrt_iswap_bd = cirq_google.PhasedFSimCharacterization(zeta=0.01, chi=0.02, gamma=0.03, theta=np.pi / 4, phi=0.0)
    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary(parameters={(a, b): {cirq.FSimGate(theta=0, phi=np.pi): params_cz_ab, cirq_google.SYC: params_syc_ab}, (c, d): {cirq.FSimGate(theta=0, phi=np.pi): params_cz_cd}, (a, c): {SQRT_ISWAP_INV_GATE: params_sqrt_iswap_ac}, (b, d): {SQRT_ISWAP_INV_GATE: params_sqrt_iswap_bd}})
    circuit = cirq.Circuit([[cirq.X(a), cirq.H(c)], [cirq.CZ.on(a, b), cirq.CZ.on(d, c)], [cirq_google.SYC.on(a, b)], [SQRT_ISWAP_GATE.on(a, c), SQRT_ISWAP_INV_GATE.on(b, d)]])
    options = cirq_google.FloquetPhasedFSimCalibrationOptions(characterize_theta=False, characterize_zeta=True, characterize_chi=True, characterize_gamma=True, characterize_phi=False)
    characterizations = [cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(theta=0, phi=np.pi), parameters={(a, b): params_cz_ab, (c, d): params_cz_cd}, options=options), cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6), parameters={(a, b): params_syc_ab}, options=options), cirq_google.PhasedFSimCalibrationResult(gate=cirq.FSimGate(theta=np.pi / 4, phi=0), parameters={(a, c): params_sqrt_iswap_ac, (b, d): params_sqrt_iswap_bd}, options=options)]
    circuit_with_calibration = workflow.make_zeta_chi_gamma_compensation_for_moments(circuit, characterizations)
    assert cirq.allclose_up_to_global_phase(engine_simulator.final_state_vector(circuit_with_calibration.circuit), cirq.final_state_vector(circuit))