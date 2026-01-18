import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_bit_flip_overload():
    d = cirq.bit_flip()
    d2 = cirq.bit_flip(0.3)
    assert str(d) == 'X'
    assert str(d2) == 'bit_flip(p=0.3)'