import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('boolean_expr,expected_pauli_sum', [('x', ['(-0.5+0j)*Z(x)', '(0.5+0j)*I']), ('~x', ['(0.5+0j)*I', '(0.5+0j)*Z(x)']), ('x0 ^ x1', ['(-0.5+0j)*Z(x0)*Z(x1)', '(0.5+0j)*I']), ('x0 & x1', ['(-0.25+0j)*Z(x0)', '(-0.25+0j)*Z(x1)', '(0.25+0j)*I', '(0.25+0j)*Z(x0)*Z(x1)']), ('x0 | x1', ['(-0.25+0j)*Z(x0)', '(-0.25+0j)*Z(x0)*Z(x1)', '(-0.25+0j)*Z(x1)', '(0.75+0j)*I']), ('x0 ^ x1 ^ x2', ['(-0.5+0j)*Z(x0)*Z(x1)*Z(x2)', '(0.5+0j)*I'])])
def test_from_boolean_expression(boolean_expr, expected_pauli_sum):
    boolean = sympy_parser.parse_expr(boolean_expr)
    qubit_map = {name: cirq.NamedQubit(name) for name in sorted(cirq.parameter_names(boolean))}
    actual = cirq.PauliSum.from_boolean_expression(boolean, qubit_map)
    actual_items = list(sorted((str(pauli_string) for pauli_string in actual)))
    assert expected_pauli_sum == actual_items