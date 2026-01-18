import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_calibration_metrics_dictionary():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    t1s = calibration['t1']
    assert t1s == {(cirq.GridQubit(0, 0),): [321], (cirq.GridQubit(0, 1),): [911], (cirq.GridQubit(1, 0),): [505]}
    assert len(calibration) == 3
    assert 't1' in calibration
    assert 't2' not in calibration
    for qubits, values in t1s.items():
        assert len(qubits) == 1
        assert len(values) == 1
    with pytest.raises(TypeError, match='was 1'):
        _ = calibration[1]
    with pytest.raises(KeyError, match='not-it'):
        _ = calibration['not-it']