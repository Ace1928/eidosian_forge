from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def check_commutation_entries(self, first_op: Operation, first_qargs: List, second_op: Operation, second_qargs: List) -> Union[bool, None]:
    """Returns stored commutation relation if any

        Args:
            first_op: first operation.
            first_qargs: first operation's qubits.
            second_op: second operation.
            second_qargs: second operation's qubits.

        Return:
            bool: True if the gates commute and false if it is not the case.
        """
    commutation = _query_commutation(first_op, first_qargs, second_op, second_qargs, self._standard_commutations)
    if commutation is not None:
        return commutation
    commutation = _query_commutation(first_op, first_qargs, second_op, second_qargs, self._cached_commutations)
    if commutation is None:
        self._cache_miss += 1
    else:
        self._cache_hit += 1
    return commutation