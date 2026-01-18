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
def test_all_operations(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(b)])])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]
    c = circuit_cls([cirq.Moment([cirq.X(a), cirq.X(b)])])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]
    c = circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(a)])])
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(a)]
    c = circuit_cls([cirq.Moment([cirq.CZ(a, b)])])
    assert list(c.all_operations()) == [cirq.CZ(a, b)]
    c = circuit_cls([cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a)])])
    assert list(c.all_operations()) == [cirq.CZ(a, b), cirq.X(a)]
    c = circuit_cls([cirq.Moment([]), cirq.Moment([cirq.X(a), cirq.Y(b)]), cirq.Moment([]), cirq.Moment([cirq.CNOT(a, b)]), cirq.Moment([cirq.Z(b), cirq.H(a)]), cirq.Moment([])])
    assert list(c.all_operations()) == [cirq.X(a), cirq.Y(b), cirq.CNOT(a, b), cirq.Z(b), cirq.H(a)]