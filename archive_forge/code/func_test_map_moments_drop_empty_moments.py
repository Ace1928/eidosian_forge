from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_map_moments_drop_empty_moments():
    op = cirq.X(cirq.NamedQubit('x'))
    c = cirq.Circuit(cirq.Moment(op), cirq.Moment(), cirq.Moment(op))
    c_mapped = cirq.map_moments(c, lambda m, i: [] if len(m) == 0 else [m])
    cirq.testing.assert_same_circuits(c_mapped, cirq.Circuit(c[0], c[0]))