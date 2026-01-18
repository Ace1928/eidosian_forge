import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_calibration_str():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    assert str(calibration) == "Calibration(keys=['globalMetric', 't1', 'two_qubit_xeb'])"