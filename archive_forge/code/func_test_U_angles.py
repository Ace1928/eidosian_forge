from typing import Callable
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
def test_U_angles():
    qasm = '\n    OPENQASM 2.0;\n    qreg q[1];\n    U(pi/2,0,pi) q[0];\n    '
    c = QasmParser().parse(qasm).circuit
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c), cirq.unitary(cirq.H), atol=1e-07)