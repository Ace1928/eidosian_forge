from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_unitary():
    m = np.diag([1, 1j])
    shape = (2, 2, 2, 2)
    rho = np.ones(shape, dtype=np.complex128)

    class HasUnitary:

        def _unitary_(self) -> np.ndarray:
            return m

    class HasUnitaryButReturnsNotImplemented(HasUnitary):

        def _apply_unitary_(self, args: cirq.ApplyMixtureArgs):
            return NotImplemented
    for val in (HasUnitary(), HasUnitaryButReturnsNotImplemented()):
        assert_apply_mixture_returns(val, rho, left_axes=[1], right_axes=[3], assert_result_is_out_buf=False, expected_result=np.reshape(np.outer([1, 1j, 1, 1j], [1, -1j, 1, -1j]), shape))