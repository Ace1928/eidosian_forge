import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_on_each_only_single_qubit():
    q0, q1 = cirq.LineQubit.range(2)
    q0_3, q1_3 = (q0.with_dimension(3), q1.with_dimension(3))
    assert cirq.I.on_each(q0, q1) == [cirq.I.on(q0), cirq.I.on(q1)]
    assert cirq.IdentityGate(1, (3,)).on_each(q0_3, q1_3) == [cirq.IdentityGate(1, (3,)).on(q0_3), cirq.IdentityGate(1, (3,)).on(q1_3)]