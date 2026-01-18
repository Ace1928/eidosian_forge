from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
def test_assert_commutes_magic_method_consistent_with_unitaries():
    gate_op = cirq.CNOT(*cirq.LineQubit.range(2))
    with pytest.raises(TypeError):
        cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(gate_op)
    exponents = [sympy.Symbol('s'), 0.1, 0.2]
    gates = [cirq.ZPowGate(exponent=e) for e in exponents]
    cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(*gates)
    cirq.testing.assert_commutes_magic_method_consistent_with_unitaries(cirq.Z, cirq.CNOT)