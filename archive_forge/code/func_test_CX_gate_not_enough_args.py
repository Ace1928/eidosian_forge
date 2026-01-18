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
def test_CX_gate_not_enough_args():
    qasm = 'OPENQASM 2.0;\n     qreg q[2];\n     CX q[0];\n'
    parser = QasmParser()
    with pytest.raises(QasmException, match='CX.*takes.*got.*1.*line 3'):
        parser.parse(qasm)