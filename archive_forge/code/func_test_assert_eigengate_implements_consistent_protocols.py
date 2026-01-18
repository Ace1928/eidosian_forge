from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
def test_assert_eigengate_implements_consistent_protocols():
    cirq.testing.assert_eigengate_implements_consistent_protocols(GoodEigenGate, global_vals={'GoodEigenGate': GoodEigenGate}, ignore_decompose_to_default_gateset=True)
    with pytest.raises(AssertionError):
        cirq.testing.assert_eigengate_implements_consistent_protocols(BadEigenGate, global_vals={'BadEigenGate': BadEigenGate}, ignore_decompose_to_default_gateset=True)