from typing import Tuple, List
from unittest.mock import create_autospec
import cirq
import numpy as np
from pyquil.gates import MEASURE, RX, DECLARE, H, CNOT, I
from pyquil.quilbase import Pragma, Reset
from cirq_rigetti import circuit_transformers as transformers
def decompose_operation(operation: cirq.Operation) -> List[cirq.Operation]:
    operations = [operation]
    if isinstance(operation.gate, cirq.MeasurementGate) and operation.gate.num_qubits() == 1:
        operations.append(cirq.I(operation.qubits[0]))
    return operations