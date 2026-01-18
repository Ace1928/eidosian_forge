import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
def test_multi_z_merges():
    q = cirq.NamedQubit('q')
    assert_optimizes(before=cirq.Circuit([cirq.Moment([cirq.Z(q) ** 0.5]), cirq.Moment([cirq.Z(q) ** 0.25])]), expected=cirq.Circuit([cirq.Moment(), cirq.Moment([cirq.Z(q) ** 0.75])]))