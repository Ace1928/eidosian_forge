import cirq
import numpy as np
import pytest
from cirq_ft.infra.decompose_protocol import (
def test_fredkin_unitary():
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    np.testing.assert_allclose(cirq.Circuit(_fredkin((c, t1, t2), context)).unitary(), cirq.unitary(cirq.FREDKIN(c, t1, t2)), atol=1e-08)