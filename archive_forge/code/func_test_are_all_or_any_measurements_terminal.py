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
def test_are_all_or_any_measurements_terminal(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    xa = cirq.X.on(a)
    xb = cirq.X.on(b)
    ma = cirq.measure(a)
    mb = cirq.measure(b)
    c = circuit_cls()
    assert c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()
    c = circuit_cls(xa, xb)
    assert c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()
    c = circuit_cls(ma)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = circuit_cls(ma, mb)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = circuit_cls(xa, ma)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = circuit_cls(xa, ma, xb, mb)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = circuit_cls(ma, xa)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()
    c = circuit_cls(ma, xa, mb)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = circuit_cls(xa, ma, xb, xa)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()
    c = circuit_cls(ma, ma)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()
    c = circuit_cls(xa, ma, xa)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()