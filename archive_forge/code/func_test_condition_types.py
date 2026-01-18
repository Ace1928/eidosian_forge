import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_condition_types():
    q0 = cirq.LineQubit(0)
    sympy_cond = sympy_parser.parse_expr('a >= 2')
    op = cirq.X(q0).with_classical_controls(cirq.MeasurementKey('a'), 'b', 'a > b', sympy_cond)
    assert set(map(str, op.classical_controls)) == {'a', 'b', 'a > b', 'a >= 2'}