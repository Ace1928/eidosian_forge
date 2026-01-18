from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def test_kraus_fallback_to_mixture():
    m = ((0.3, cirq.unitary(cirq.X)), (0.4, cirq.unitary(cirq.Y)), (0.3, cirq.unitary(cirq.Z)))

    class ReturnsMixture:

        def _mixture_(self) -> Iterable[Tuple[float, np.ndarray]]:
            return m
    c = (np.sqrt(0.3) * cirq.unitary(cirq.X), np.sqrt(0.4) * cirq.unitary(cirq.Y), np.sqrt(0.3) * cirq.unitary(cirq.Z))
    np.testing.assert_equal(cirq.kraus(ReturnsMixture()), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), None), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), NotImplemented), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), (1,)), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), LOCAL_DEFAULT), c)
    assert cirq.has_kraus(ReturnsMixture())