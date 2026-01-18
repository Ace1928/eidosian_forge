from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('exponent', (-5.5, -3, -1.5, -1, -0.65, -0.2, 0, 0.1, 0.75, 1, 1.5, 2, 5.5))
def test_decomposition_to_sycamore_gate(exponent):
    cphase_gate = cirq.CZPowGate(exponent=exponent)
    assert_decomposition_valid(cphase_gate, FakeSycamoreGate())