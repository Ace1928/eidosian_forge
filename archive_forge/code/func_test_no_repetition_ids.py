import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_no_repetition_ids():

    def default_repetition_ids(self):
        assert False, 'Should not call default_repetition_ids'
    with mock.patch.object(circuit_operation, 'default_repetition_ids', new=default_repetition_ids):
        q = cirq.LineQubit(0)
        op = cirq.CircuitOperation(cirq.Circuit(cirq.X(q), cirq.measure(q)).freeze(), repetitions=1000000, use_repetition_ids=False)
        assert op.repetitions == 1000000
        assert op.repetition_ids is None
        _ = repr(op)
        _ = str(op)
        op2 = op.repeat(10)
        assert op2.repetitions == 10000000
        assert op2.repetition_ids is None