import functools
import itertools
import math
import random
import numpy as np
import pytest
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.ops.boolean_hamiltonian as bh
def test_gate_consistent():
    gate = cirq.BooleanHamiltonianGate(['a', 'b'], ['a'], 0.1)
    op = gate.on(*cirq.LineQubit.range(2))
    cirq.testing.assert_implements_consistent_protocols(gate)
    cirq.testing.assert_implements_consistent_protocols(op)