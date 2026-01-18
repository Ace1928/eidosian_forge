from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def merge_func(op1, op2):
    return cirq.I(*op1.qubits) if op1 == measure_op and op2 == measure_op else None