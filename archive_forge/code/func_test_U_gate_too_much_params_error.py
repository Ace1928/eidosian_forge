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
def test_U_gate_too_much_params_error():
    qasm = 'OPENQASM 2.0;\n     qreg q[2];     \n     U(pi, pi, pi, pi) q[1];'
    parser = QasmParser()
    with pytest.raises(QasmException, match='U takes 3.*got.*4.*line 3'):
        parser.parse(qasm)