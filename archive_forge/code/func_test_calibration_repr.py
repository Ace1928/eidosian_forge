import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_calibration_repr():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    cirq.testing.assert_equivalent_repr(calibration, setup_code='import cirq\nimport cirq_google')