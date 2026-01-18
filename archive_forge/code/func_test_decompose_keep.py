import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
def test_decompose_keep():
    a, b = cirq.LineQubit.range(2)
    assert cirq.decompose(cirq.SWAP(a, b), keep=lambda e: isinstance(e.gate, cirq.CNotPowGate)) == [cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.CNOT(a, b)]
    cirq.testing.assert_has_diagram(cirq.Circuit(cirq.decompose(cirq.SWAP(a, b))), '\n0: ────────────@───Y^-0.5───@───Y^0.5────@───────────\n               │            │            │\n1: ───Y^-0.5───@───Y^0.5────@───Y^-0.5───@───Y^0.5───\n')
    assert cirq.decompose(cirq.SWAP(a, b), keep=lambda _: True) == [cirq.SWAP(a, b)]
    assert cirq.decompose(DecomposeGiven(cirq.SWAP(b, a)), keep=lambda _: True) == [cirq.SWAP(b, a)]
    assert cirq.decompose([[[cirq.SWAP(a, b)]]], keep=lambda _: True) == [cirq.SWAP(a, b)]