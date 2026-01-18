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
def test_measure_from_classical_register():
    qasm = 'OPENQASM 2.0;\n         include "qelib1.inc";       \n         qreg q1[2];\n         creg c1[3];                        \n         creg c2[3];                        \n         measure c1 -> c2;       \n    '
    parser = QasmParser()
    with pytest.raises(QasmException, match='Undefined quantum register.*c1.*line 6'):
        parser.parse(qasm)