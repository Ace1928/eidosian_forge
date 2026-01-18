import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def mixtures_equal(m1, m2, atol=1e-07):
    for (p1, v1), (p2, v2) in zip(m1, m2):
        if not (cirq.approx_eq(p1, p2, atol=atol) and cirq.equal_up_to_global_phase(v1, v2, atol=atol)):
            return False
    return True