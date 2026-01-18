import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
def test_merge_matching_results_when_incompatible_fails():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    parameters_1 = {(q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3)}
    parameters_2 = {(q_02, q_03): PhasedFSimCharacterization(theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6)}
    with pytest.raises(ValueError):
        results = [PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options), PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options)]
        assert merge_matching_results(results)
    with pytest.raises(ValueError):
        results = [PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options), PhasedFSimCalibrationResult(parameters=parameters_2, gate=cirq.CZ, options=options)]
        assert merge_matching_results(results)
    with pytest.raises(ValueError):
        results = [PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options), PhasedFSimCalibrationResult(parameters=parameters_2, gate=gate, options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION)]
        assert merge_matching_results(results)