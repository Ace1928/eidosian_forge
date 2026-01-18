from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def commute(self, op1: Operation, qargs1: List, cargs1: List, op2: Operation, qargs2: List, cargs2: List, max_num_qubits: int=3) -> bool:
    """
        Checks if two Operations commute. The return value of `True` means that the operations
        truly commute, and the return value of `False` means that either the operations do not
        commute or that the commutation check was skipped (for example, when the operations
        have conditions or have too many qubits).

        Args:
            op1: first operation.
            qargs1: first operation's qubits.
            cargs1: first operation's clbits.
            op2: second operation.
            qargs2: second operation's qubits.
            cargs2: second operation's clbits.
            max_num_qubits: the maximum number of qubits to consider, the check may be skipped if
                the number of qubits for either operation exceeds this amount.

        Returns:
            bool: whether two operations commute.
        """
    structural_commutation = _commutation_precheck(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits)
    if structural_commutation is not None:
        return structural_commutation
    first_op_tuple, second_op_tuple = _order_operations(op1, qargs1, cargs1, op2, qargs2, cargs2)
    first_op, first_qargs, _ = first_op_tuple
    second_op, second_qargs, _ = second_op_tuple
    first_params = first_op.params
    second_params = second_op.params
    commutation_lookup = self.check_commutation_entries(first_op, first_qargs, second_op, second_qargs)
    if commutation_lookup is not None:
        return commutation_lookup
    is_commuting = _commute_matmul(first_op, first_qargs, second_op, second_qargs)
    if self._current_cache_entries >= self._cache_max_entries:
        self.clear_cached_commutations()
    if len(first_params) > 0 or len(second_params) > 0:
        self._cached_commutations.setdefault((first_op.name, second_op.name), {}).setdefault(_get_relative_placement(first_qargs, second_qargs), {})[_hashable_parameters(first_params), _hashable_parameters(second_params)] = is_commuting
    else:
        self._cached_commutations.setdefault((first_op.name, second_op.name), {})[_get_relative_placement(first_qargs, second_qargs)] = is_commuting
    self._current_cache_entries += 1
    return is_commuting