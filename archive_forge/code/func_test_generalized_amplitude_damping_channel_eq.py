import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_generalized_amplitude_damping_channel_eq():
    a = cirq.generalized_amplitude_damp(0.0099999, 0.01)
    b = cirq.generalized_amplitude_damp(0.01, 0.0099999)
    assert cirq.approx_eq(a, b, atol=0.01)
    et = cirq.testing.EqualsTester()
    c = cirq.generalized_amplitude_damp(0.0, 0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.generalized_amplitude_damp(0.1, 0.0))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.0, 0.1))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.6, 0.4))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.8, 0.2))