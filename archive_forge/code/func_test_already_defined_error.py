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
@pytest.mark.parametrize('qasm', ['OPENQASM 2.0;\n           qreg q[2];\n           creg q[3];\n               ', 'OPENQASM 2.0;\n           creg q[2];\n           qreg q[3];\n               '])
def test_already_defined_error(qasm: str):
    parser = QasmParser()
    with pytest.raises(QasmException, match='q.*already defined.* line 3'):
        parser.parse(qasm)