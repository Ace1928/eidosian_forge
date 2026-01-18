import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_state_vector_ignore_terminal_measurement():
    a, b = cirq.LineQubit.range(2)
    np.testing.assert_allclose(cirq.final_state_vector([cirq.X(a), cirq.X(b) ** 0.5, cirq.measure(a, b, key='m')], ignore_terminal_measurements=True), [0, 0, 0.5 + 0.5j, 0.5 - 0.5j])
    with pytest.raises(ValueError, match='is not unitary'):
        _ = (cirq.final_state_vector([cirq.X(a), cirq.amplitude_damp(0.1).on(b), cirq.measure(a, b, key='m')], ignore_terminal_measurements=True),)