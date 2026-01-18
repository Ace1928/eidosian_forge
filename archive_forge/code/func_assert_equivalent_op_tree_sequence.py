from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def assert_equivalent_op_tree_sequence(x: Sequence[cirq.OP_TREE], y: Sequence[cirq.OP_TREE]):
    assert len(x) == len(y)
    for a, b in zip(x, y):
        assert_equivalent_op_tree(a, b)