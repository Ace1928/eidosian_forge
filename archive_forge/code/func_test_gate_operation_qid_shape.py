import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_operation_qid_shape():

    class ShapeGate(cirq.Gate):

        def _qid_shape_(self):
            return (1, 2, 3, 4)
    op = ShapeGate().on(*cirq.LineQid.for_qid_shape((1, 2, 3, 4)))
    assert cirq.qid_shape(op) == (1, 2, 3, 4)
    assert cirq.num_qubits(op) == 4