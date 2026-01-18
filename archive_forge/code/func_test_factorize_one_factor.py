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
def test_factorize_one_factor():
    circuit = cirq.Circuit()
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit.append([cirq.Moment([cirq.CZ(q0, q1), cirq.H(q2)]), cirq.Moment([cirq.H(q0), cirq.CZ(q1, q2)])])
    factors = list(circuit.factorize())
    assert len(factors) == 1
    assert factors[0] == circuit
    desired = '\n0: ───@───H───\n      │\n1: ───@───@───\n          │\n2: ───H───@───\n'
    cirq.testing.assert_has_diagram(factors[0], desired)