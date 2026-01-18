import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_asymmetric_depolarizing_channel_eq():
    a = cirq.asymmetric_depolarize(0.0099999, 0.01)
    b = cirq.asymmetric_depolarize(0.01, 0.0099999)
    c = cirq.asymmetric_depolarize(0.0, 0.0, 0.0)
    assert cirq.approx_eq(a, b, atol=0.01)
    et = cirq.testing.EqualsTester()
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.0, 0.1))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.1, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.1, 0.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.1, 0.2, 0.3))
    et.add_equality_group(cirq.asymmetric_depolarize(0.3, 0.4, 0.3))
    et.add_equality_group(cirq.asymmetric_depolarize(1.0, 0.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 1.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.0, 1.0))