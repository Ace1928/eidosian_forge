import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_global():
    qubits = cirq.LineQubit.range(3)
    assert cirq.identity_each(*qubits) == cirq.IdentityGate(3).on(*qubits)
    qids = cirq.LineQid.for_qid_shape((1, 2, 3))
    assert cirq.identity_each(*qids) == cirq.IdentityGate(3, (1, 2, 3)).on(*qids)
    with pytest.raises(ValueError, match='Not a cirq.Qid'):
        cirq.identity_each(qubits)