import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_controlled_operation_eq():
    c1 = cirq.NamedQubit('c1')
    q1 = cirq.NamedQubit('q1')
    c2 = cirq.NamedQubit('c2')
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.ControlledOperation([c1], cirq.X(q1)))
    eq.make_equality_group(lambda: cirq.ControlledOperation([c2], cirq.X(q1)))
    eq.make_equality_group(lambda: cirq.ControlledOperation([c1], cirq.Z(q1)))
    eq.add_equality_group(cirq.ControlledOperation([c2], cirq.Z(q1)))
    eq.add_equality_group(cirq.ControlledOperation([c1, c2], cirq.Z(q1)), cirq.ControlledOperation([c2, c1], cirq.Z(q1)))
    eq.add_equality_group(cirq.ControlledOperation([c1, c2.with_dimension(3)], cirq.Z(q1), control_values=[1, (0, 2)]), cirq.ControlledOperation([c2.with_dimension(3), c1], cirq.Z(q1), control_values=[(2, 0), 1]))