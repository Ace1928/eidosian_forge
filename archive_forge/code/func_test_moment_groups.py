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
def test_moment_groups(circuit_cls):
    qubits = [cirq.GridQubit(x, y) for x in range(8) for y in range(8)]
    c0 = cirq.H(qubits[0])
    c7 = cirq.H(qubits[7])
    cz14 = cirq.CZ(qubits[1], qubits[4])
    cz25 = cirq.CZ(qubits[2], qubits[5])
    cz36 = cirq.CZ(qubits[3], qubits[6])
    moment1 = cirq.Moment([c0, cz14, cz25, c7])
    moment2 = cirq.Moment([c0, cz14, cz25, cz36, c7])
    moment3 = cirq.Moment([cz14, cz25, cz36])
    moment4 = cirq.Moment([cz25, cz36])
    circuit = circuit_cls((moment1, moment2, moment3, moment4))
    cirq.testing.assert_has_diagram(circuit, '\n           ┌──┐   ┌───┐   ┌───┐   ┌──┐\n(0, 0): ────H──────H─────────────────────\n\n(0, 1): ────@──────@───────@─────────────\n            │      │       │\n(0, 2): ────┼@─────┼@──────┼@──────@─────\n            ││     ││      ││      │\n(0, 3): ────┼┼─────┼┼@─────┼┼@─────┼@────\n            ││     │││     │││     ││\n(0, 4): ────@┼─────@┼┼─────@┼┼─────┼┼────\n             │      ││      ││     ││\n(0, 5): ─────@──────@┼──────@┼─────@┼────\n                     │       │      │\n(0, 6): ─────────────@───────@──────@────\n\n(0, 7): ────H──────H─────────────────────\n           └──┘   └───┘   └───┘   └──┘\n', use_unicode_characters=True)