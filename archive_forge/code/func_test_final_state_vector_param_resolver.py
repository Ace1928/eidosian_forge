import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_state_vector_param_resolver():
    s = sympy.Symbol('s')
    with pytest.raises(ValueError, match='not unitary'):
        _ = cirq.final_state_vector(cirq.X ** s)
    np.testing.assert_allclose(cirq.final_state_vector(cirq.X ** s, param_resolver={s: 0.5}), [0.5 + 0.5j, 0.5 - 0.5j])