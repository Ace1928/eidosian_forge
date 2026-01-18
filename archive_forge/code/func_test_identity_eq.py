import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.make_equality_group(lambda: cirq.I, lambda: cirq.IdentityGate(1), lambda: cirq.IdentityGate(1, (2,)))
    equals_tester.add_equality_group(cirq.IdentityGate(2), cirq.IdentityGate(2, (2, 2)))
    equals_tester.add_equality_group(cirq.IdentityGate(4))
    equals_tester.add_equality_group(cirq.IdentityGate(1, (3,)))
    equals_tester.add_equality_group(cirq.IdentityGate(4, (1, 2, 3, 4)))