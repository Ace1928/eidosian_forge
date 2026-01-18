import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_fails_on_big_unknowns():

    class UnrecognizedGate(cirq.testing.ThreeQubitGate):
        pass
    c = cirq.Circuit(UnrecognizedGate().on(*cirq.LineQubit.range(3)))
    with pytest.raises(ValueError, match='Cannot output operation as QASM'):
        _ = c.to_qasm()