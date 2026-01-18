import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_bit_flip_channel_str():
    assert str(cirq.bit_flip(0.3)) == 'bit_flip(p=0.3)'