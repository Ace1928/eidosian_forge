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
def test_findall_operations(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    xa = cirq.X.on(a)
    xb = cirq.X.on(b)
    za = cirq.Z.on(a)
    zb = cirq.Z.on(b)

    def is_x(op: cirq.Operation) -> bool:
        return isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.XPowGate)
    c = circuit_cls()
    assert list(c.findall_operations(is_x)) == []
    c = circuit_cls(xa)
    assert list(c.findall_operations(is_x)) == [(0, xa)]
    c = circuit_cls(za)
    assert list(c.findall_operations(is_x)) == []
    c = circuit_cls([za, zb] * 8)
    assert list(c.findall_operations(is_x)) == []
    c = circuit_cls(xa, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (0, xb)]
    c = circuit_cls(xa, zb)
    assert list(c.findall_operations(is_x)) == [(0, xa)]
    c = circuit_cls(xa, za)
    assert list(c.findall_operations(is_x)) == [(0, xa)]
    c = circuit_cls([xa] * 8)
    assert list(c.findall_operations(is_x)) == list(enumerate([xa] * 8))
    c = circuit_cls(za, zb, xa, xb)
    assert list(c.findall_operations(is_x)) == [(1, xa), (1, xb)]
    c = circuit_cls(xa, zb, za, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (1, xb)]