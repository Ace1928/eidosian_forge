import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_operation_num_qubits():

    class NumQubitsGate(cirq.Gate):

        def _num_qubits_(self):
            return 4
    op = NumQubitsGate().on(*cirq.LineQubit.range(4))
    assert cirq.qid_shape(op) == (2, 2, 2, 2)
    assert cirq.num_qubits(op) == 4