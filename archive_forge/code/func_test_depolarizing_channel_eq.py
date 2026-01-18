import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel_eq():
    a = cirq.depolarize(p=0.0099999)
    b = cirq.depolarize(p=0.01)
    c = cirq.depolarize(0.0)
    assert cirq.approx_eq(a, b, atol=0.01)
    et = cirq.testing.EqualsTester()
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.depolarize(0.1))
    et.add_equality_group(cirq.depolarize(0.9))
    et.add_equality_group(cirq.depolarize(1.0))