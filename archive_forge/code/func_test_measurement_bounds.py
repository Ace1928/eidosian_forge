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
def test_measurement_bounds():
    qasm = 'OPENQASM 2.0;\n     qreg q1[3];\n     creg c1[3];                        \n     measure q1[0] -> c1[4];  \n'
    parser = QasmParser()
    with pytest.raises(QasmException, match='Out of bounds bit.*4.*c1.*size 3.*line 4'):
        parser.parse(qasm)