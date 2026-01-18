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
def test_floquet_parse_result():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(gate=gate, pairs=((q_00, q_01), (q_02, q_03)), options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))
    result = cirq_google.CalibrationResult(code=cirq_google.api.v2.calibration_pb2.SUCCESS, error_message=None, token=None, valid_until=None, metrics=cirq_google.Calibration(cirq_google.api.v2.metrics_pb2.MetricsSnapshot(metrics=[cirq_google.api.v2.metrics_pb2.Metric(name='angles', targets=['0_qubit_a', '0_qubit_b', '0_theta_est', '0_zeta_est', '0_phi_est', '1_qubit_a', '1_qubit_b', '1_theta_est', '1_zeta_est', '1_phi_est'], values=[cirq_google.api.v2.metrics_pb2.Value(str_val='0_0'), cirq_google.api.v2.metrics_pb2.Value(str_val='0_1'), cirq_google.api.v2.metrics_pb2.Value(double_val=0.1), cirq_google.api.v2.metrics_pb2.Value(double_val=0.2), cirq_google.api.v2.metrics_pb2.Value(double_val=0.3), cirq_google.api.v2.metrics_pb2.Value(str_val='0_2'), cirq_google.api.v2.metrics_pb2.Value(str_val='0_3'), cirq_google.api.v2.metrics_pb2.Value(double_val=0.4), cirq_google.api.v2.metrics_pb2.Value(double_val=0.5), cirq_google.api.v2.metrics_pb2.Value(double_val=0.6)])])))
    assert request.parse_result(result) == PhasedFSimCalibrationResult(parameters={(q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3), (q_02, q_03): PhasedFSimCharacterization(theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6)}, gate=gate, options=FloquetPhasedFSimCalibrationOptions(characterize_theta=True, characterize_zeta=True, characterize_chi=False, characterize_gamma=False, characterize_phi=True))