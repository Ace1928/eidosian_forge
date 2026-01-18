import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_hqp_equality():
    hqp = cg.HardcodedQubitPlacer({cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(5)))})
    hqp2 = cg.HardcodedQubitPlacer({cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(5)))})
    assert hqp == hqp2
    cirq.testing.assert_equivalent_repr(hqp, global_vals={'cirq_google': cg})
    hqp3 = cg.HardcodedQubitPlacer({cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(1, 5 + 1)))})
    assert hqp != hqp3