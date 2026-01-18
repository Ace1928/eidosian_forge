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
def test_diagram_global_phase(circuit_cls):
    qa = cirq.NamedQubit('a')
    global_phase = cirq.global_phase_operation(coefficient=1j)
    c = circuit_cls([global_phase])
    cirq.testing.assert_has_diagram(c, '\n\nglobal phase:   0.5pi', use_unicode_characters=False, precision=2)
    cirq.testing.assert_has_diagram(c, '\n\nglobal phase:   0.5π', use_unicode_characters=True, precision=2)
    c = circuit_cls([cirq.X(qa), global_phase, global_phase])
    cirq.testing.assert_has_diagram(c, 'a: ─────────────X───\n\nglobal phase:   π', use_unicode_characters=True, precision=2)
    c = circuit_cls([cirq.X(qa), global_phase], cirq.Moment([cirq.X(qa), global_phase]))
    cirq.testing.assert_has_diagram(c, 'a: ─────────────X──────X──────\n\nglobal phase:   0.5π   0.5π\n', use_unicode_characters=True, precision=2)
    c = circuit_cls(cirq.X(cirq.LineQubit(2)), cirq.CircuitOperation(circuit_cls(cirq.global_phase_operation(-1).with_tags('tag')).freeze()))
    cirq.testing.assert_has_diagram(c, "2: ───X──────────\n\n      π['tag']")