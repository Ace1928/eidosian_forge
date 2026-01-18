from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_simple_split_fallback():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasMixture:

        def _mixture_(self):
            return ((0.5, np.eye(2, dtype=np.complex128)), (0.5, cirq.X))
    rho = np.copy(x)
    assert_apply_mixture_returns(HasMixture(), rho, [0], [1], assert_result_is_out_buf=True, expected_result=x)