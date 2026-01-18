from math import exp
import pytest
from google.protobuf.text_format import Merge
import cirq
from cirq.testing import assert_equivalent_op_tree
import cirq_google
from cirq_google.api import v2
from cirq_google.experimental.noise_models import simple_noise_from_calibration_metrics
def test_noise_from_metrics_unsupported():
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    with pytest.raises(NotImplementedError, match='Gate damping is not yet supported.'):
        simple_noise_from_calibration_metrics(calibration=calibration, damping_noise=True)