import random
import numpy as np
import pytest
import cirq
def assert_ms_depth_below(operations, threshold):
    total_ms = 0
    for op in operations:
        assert len(op.qubits) <= 2
        if len(op.qubits) == 2:
            assert isinstance(op, cirq.GateOperation)
            assert isinstance(op.gate, cirq.XXPowGate)
            total_ms += abs(op.gate.exponent)
    assert total_ms <= threshold