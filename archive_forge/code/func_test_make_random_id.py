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
def test_make_random_id():
    with mock.patch('random.choice', return_value='A'):
        random_id = cg.engine.engine._make_random_id('prefix-', length=4)
        assert random_id[:11] == 'prefix-AAAA'
    random_id = cg.engine.engine._make_random_id('prefix-')
    time.sleep(1)
    random_id2 = cg.engine.engine._make_random_id('prefix-')
    assert random_id[-7:] != '-000000' or random_id2[-7:] != '-000000'
    assert random_id != random_id2