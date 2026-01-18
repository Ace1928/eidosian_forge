from datetime import timedelta
from typing import List
import pytest
import cirq
from cirq_aqt import aqt_device, aqt_device_metadata
def test_aqt_device_eq(device):
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: device)