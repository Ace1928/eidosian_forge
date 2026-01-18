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
def test_circuit_unitary(circuit_cls):
    q = cirq.NamedQubit('q')
    with_inner_measure = circuit_cls(cirq.H(q), cirq.measure(q), cirq.H(q))
    assert not cirq.has_unitary(with_inner_measure)
    assert cirq.unitary(with_inner_measure, None) is None
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(circuit_cls(cirq.X(q) ** 0.5), cirq.measure(q)), np.array([[1j, 1], [1, 1j]]) * np.sqrt(0.5), atol=1e-08)