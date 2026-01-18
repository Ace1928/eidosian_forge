import numpy as np
import pytest
import sympy
import cirq
def assert_channel_sums_to_identity(val):
    m = cirq.kraus(val)
    s = sum((np.conj(e.T) @ e for e in m))
    np.testing.assert_allclose(s, np.eye(np.prod(cirq.qid_shape(val), dtype=np.int64)), atol=1e-08)