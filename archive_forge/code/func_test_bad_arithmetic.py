import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_bad_arithmetic():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum = pstr1 + 2 * pstr2 + 1
    with pytest.raises(TypeError):
        psum += 'hi mom'
    with pytest.raises(TypeError):
        _ = psum + 'hi mom'
    with pytest.raises(TypeError):
        psum -= 'hi mom'
    with pytest.raises(TypeError):
        _ = psum - 'hi mom'
    with pytest.raises(TypeError):
        psum *= [1, 2, 3]
    with pytest.raises(TypeError):
        _ = psum * [1, 2, 3]
    with pytest.raises(TypeError):
        _ = [1, 2, 3] * psum
    with pytest.raises(TypeError):
        _ = psum / [1, 2, 3]
    with pytest.raises(TypeError):
        _ = psum ** 1.2
    with pytest.raises(TypeError):
        _ = psum ** (-2)
    with pytest.raises(TypeError):
        _ = psum ** 'string'