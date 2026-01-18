from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_moments_empty_circuit():

    def fail_if_called_func(*_):
        assert False
    c = cirq.Circuit()
    assert cirq.merge_moments(c, fail_if_called_func) is c