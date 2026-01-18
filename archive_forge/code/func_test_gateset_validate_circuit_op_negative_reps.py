from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_gateset_validate_circuit_op_negative_reps():
    gate = CustomXPowGate(exponent=0.5)
    op = cirq.CircuitOperation(cirq.FrozenCircuit(gate.on(cirq.LineQubit(0))), repetitions=-1)
    assert op not in cirq.Gateset(gate)
    assert op ** (-1) in cirq.Gateset(gate)