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
def test_init_contents(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    c = circuit_cls(cirq.Moment([cirq.H(a)]), cirq.Moment([cirq.X(b)]), cirq.Moment([cirq.CNOT(a, b)]))
    assert len(c.moments) == 3
    c = circuit_cls(cirq.H(a), cirq.X(b), cirq.CNOT(a, b))
    assert c == circuit_cls(cirq.Moment([cirq.H(a), cirq.X(b)]), cirq.Moment([cirq.CNOT(a, b)]))
    c = circuit_cls(cirq.H(a), cirq.X(b), cirq.CNOT(a, b), strategy=cirq.InsertStrategy.NEW)
    assert c == circuit_cls(cirq.Moment([cirq.H(a)]), cirq.Moment([cirq.X(b)]), cirq.Moment([cirq.CNOT(a, b)]))
    circuit_cls()