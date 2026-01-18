import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_controlled_diagram_exponent():
    for q in itertools.permutations(cirq.LineQubit.range(5)):
        for idx in [None, 0, 1]:
            op = MockGate(idx)(*q[:2]).controlled_by(*q[2:])
            add = 0 if idx is None else idx
            assert cirq.circuit_diagram_info(op).exponent_qubit_index == len(q[2:]) + add