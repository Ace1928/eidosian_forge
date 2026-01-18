import fractions
import numpy as np
import pytest
import sympy
import cirq
def test_formulas_in_param_dict():
    """Tests that formula keys are rejected in a `param_dict`."""
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    e = sympy.Symbol('e')
    with pytest.raises(TypeError, match='formula'):
        _ = cirq.ParamResolver({a: b + 1, b: 2, b + c: 101, 'd': 2 * e})