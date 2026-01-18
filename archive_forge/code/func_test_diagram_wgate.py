import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_diagram_wgate(circuit_cls):
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.PhasedXPowGate(exponent=0.12341234, phase_exponent=0.43214321)
    c = circuit_cls([cirq.Moment([test_wgate.on(qa)])])
    cirq.testing.assert_has_diagram(c, '\na: ---PhX(0.43)^(1/8)---\n', use_unicode_characters=False, precision=2)