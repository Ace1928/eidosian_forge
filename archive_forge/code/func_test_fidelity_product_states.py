import numpy as np
import pytest
import cirq
def test_fidelity_product_states():
    a, b = cirq.LineQubit.range(2)
    np.testing.assert_allclose(cirq.fidelity(cirq.KET_ZERO(a) * cirq.KET_ZERO(b), cirq.KET_ZERO(a) * cirq.KET_ZERO(b)), 1.0)
    np.testing.assert_allclose(cirq.fidelity(cirq.KET_ZERO(a) * cirq.KET_ZERO(b), cirq.KET_ZERO(a) * cirq.KET_ONE(b)), 0.0, atol=1e-07)
    np.testing.assert_allclose(cirq.fidelity(cirq.KET_ZERO(a) * cirq.KET_ZERO(b), cirq.KET_ZERO(a) * cirq.KET_PLUS(b)), 0.5)
    np.testing.assert_allclose(cirq.fidelity(cirq.KET_ONE(a) * cirq.KET_ONE(b), cirq.KET_MINUS(a) * cirq.KET_PLUS(b)), 0.25)
    np.testing.assert_allclose(cirq.fidelity(cirq.KET_MINUS(a) * cirq.KET_PLUS(b), cirq.KET_MINUS(a) * cirq.KET_PLUS(b)), 1.0)
    np.testing.assert_allclose(cirq.fidelity(cirq.KET_MINUS(a) * cirq.KET_PLUS(b), cirq.KET_PLUS(a) * cirq.KET_MINUS(b)), 0.0, atol=1e-07)
    with pytest.raises(ValueError, match='Mismatched'):
        _ = cirq.fidelity(cirq.KET_MINUS(a), cirq.KET_PLUS(a) * cirq.KET_MINUS(b))
    with pytest.raises(ValueError, match='qid shape'):
        _ = cirq.fidelity(cirq.KET_MINUS(a) * cirq.KET_PLUS(b), cirq.KET_PLUS(a) * cirq.KET_MINUS(b), qid_shape=(4,))