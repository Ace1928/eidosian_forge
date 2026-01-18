import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_condition_flattening():
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_classical_controls('a').with_classical_controls('b')
    assert set(map(str, op.classical_controls)) == {'a', 'b'}
    assert isinstance(op._sub_operation, cirq.GateOperation)