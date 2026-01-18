import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_current_calibration')
def test_get_engine_calibration(get_current_calibration):
    get_current_calibration.return_value = _CALIBRATION
    calibration = cirq_google.get_engine_calibration('rainbow', 'project')
    assert calibration.timestamp == 1562544000021
    assert set(calibration.keys()) == {'t1', 'globalMetric'}
    assert calibration['t1'][cirq.GridQubit(0, 0),] == [321.0]
    get_current_calibration.assert_called_once_with('project', 'rainbow')