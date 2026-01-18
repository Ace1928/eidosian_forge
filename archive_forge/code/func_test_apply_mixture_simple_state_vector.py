from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_simple_state_vector():
    for _ in range(25):
        state = cirq.testing.random_superposition(2)
        u1 = cirq.testing.random_unitary(2)
        u2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        p1 = np.random.random()
        p2 = 1 - p1
        expected = p1 * np.dot(u1, state) + p2 * np.dot(u2, state)

        class HasMixture:

            def _mixture_(self):
                return ((p1, u1), (p2, cirq.X))
        assert_apply_mixture_returns(HasMixture(), state, [0], None, assert_result_is_out_buf=True, expected_result=expected)