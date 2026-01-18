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
def test_CX_gate_arg_overlap():
    qasm = 'OPENQASM 2.0;\n     qreg q1[2];\n     qreg q2[3];\n     CX q1[1], q1[1];\n'
    parser = QasmParser()
    with pytest.raises(QasmException, match='Overlapping.*at line 4'):
        parser.parse(qasm)