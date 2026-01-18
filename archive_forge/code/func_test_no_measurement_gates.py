import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_no_measurement_gates():
    q0 = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='with measurements'):
        _ = cirq.measure(q0).with_classical_controls('a')