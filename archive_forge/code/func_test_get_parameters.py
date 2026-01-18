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
def test_get_parameters():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    result = PhasedFSimCalibrationResult(parameters={(q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3), (q_02, q_03): PhasedFSimCharacterization(theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6)}, gate=gate, options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))
    assert result.get_parameters(q_00, q_01) == PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3)
    assert result.get_parameters(q_01, q_00) == PhasedFSimCharacterization(theta=0.1, zeta=-0.2, chi=None, gamma=None, phi=0.3)
    assert result.get_parameters(q_02, q_03) == PhasedFSimCharacterization(theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6)
    assert result.get_parameters(q_00, q_03) is None