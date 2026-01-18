import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_empty_circuit_one_qubit():
    q0, = _make_qubits(1)
    output = cirq.QasmOutput((), (q0,))
    assert str(output) == 'OPENQASM 2.0;\ninclude "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n'