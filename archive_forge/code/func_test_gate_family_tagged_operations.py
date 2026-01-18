from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('gate_family, gates_to_check', [(cirq.GateFamily(cirq.ZPowGate, tags_to_accept=['a', 'b']), [(cirq.Z(q).with_tags('a', 'b'), True), (cirq.Z(q).with_tags('a'), True), (cirq.Z(q).with_tags('b'), True), (cirq.Z(q).with_tags('c'), False), (cirq.Z(q).with_tags('a', 'c'), True), (cirq.Z(q).with_tags(), False), (cirq.Z(q), False), (cirq.Z, False), (cirq.X(q).with_tags('a'), False), (cirq.X(q).with_tags('c'), False)]), (cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=['a', 'b']), [(cirq.Z(q).with_tags('a', 'b'), False), (cirq.Z(q).with_tags('a'), False), (cirq.Z(q).with_tags('b'), False), (cirq.Z(q).with_tags('c'), True), (cirq.Z(q).with_tags('a', 'c'), False), (cirq.Z(q).with_tags(), True), (cirq.Z(q), True), (cirq.Z, True), (cirq.X(q).with_tags('a'), False), (cirq.X(q).with_tags('c'), False)]), (cirq.GateFamily(cirq.ZPowGate, tags_to_accept=['a'], tags_to_ignore=['c']), [(cirq.Z(q).with_tags('a', 'c'), False), (cirq.Z(q).with_tags('a'), True), (cirq.Z(q).with_tags('c'), False), (cirq.Z(q).with_tags(), False), (cirq.Z(q), False), (cirq.Z, False), (cirq.X(q).with_tags('a'), False), (cirq.X(q).with_tags('c'), False)])])
def test_gate_family_tagged_operations(gate_family, gates_to_check):
    for gate, result in gates_to_check:
        assert (gate in gate_family) == result