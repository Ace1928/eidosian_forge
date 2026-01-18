import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_without_context_succeed():
    op = G2()(cirq.NamedQubit('q'))
    assert cirq.decompose(op, keep=lambda op: op.gate is cirq.CNOT) == [cirq.CNOT(cirq.ops.CleanQubit(0, prefix='_decompose_protocol'), cirq.ops.CleanQubit(1, prefix='_decompose_protocol'))]