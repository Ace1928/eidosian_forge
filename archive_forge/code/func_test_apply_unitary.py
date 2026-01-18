import itertools
import numpy as np
import pytest
import cirq
def test_apply_unitary():
    cirq.testing.assert_has_consistent_apply_unitary(cirq.X)
    cirq.testing.assert_has_consistent_apply_unitary(cirq.Y)
    cirq.testing.assert_has_consistent_apply_unitary(cirq.Z)