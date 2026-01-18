import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
def get_crosstalk_operation(self, operation: cirq.Operation, system_qubits: Sequence[cirq.Qid]) -> List[cirq.Operation]:
    """Returns a list of operations including crosstalk

        Args:
            operation: Ideal operation
            system_qubits: Tuple of line qubits

        Returns:
            List of operations including crosstalk
        """
    cast(Tuple[cirq.LineQubit], system_qubits)
    num_qubits = len(system_qubits)
    xtlk_arr = np.zeros(num_qubits)
    idx_list = []
    for qubit in operation.qubits:
        idx = system_qubits.index(qubit)
        idx_list.append(idx)
        neighbors = [idx - 1, idx + 1]
        for neigh_idx in neighbors:
            if neigh_idx >= 0 and neigh_idx < num_qubits:
                xtlk_arr[neigh_idx] = self.noise_op_dict['crosstalk']
    for idx in idx_list:
        xtlk_arr[idx] = 0
    xtlk_op_list = []
    op_str = get_op_string(operation)
    gate = cast(cirq.EigenGate, gate_dict[op_str])
    if len(operation.qubits) == 1:
        for idx in xtlk_arr.nonzero()[0]:
            exponent = operation.gate.exponent
            exponent = exponent * xtlk_arr[idx]
            xtlk_op = gate.on(system_qubits[idx]) ** exponent
            xtlk_op_list.append(xtlk_op)
    elif len(operation.qubits) == 2:
        for op_qubit in operation.qubits:
            for idx in xtlk_arr.nonzero()[0]:
                exponent = operation.gate.exponent
                exponent = exponent * xtlk_arr[idx]
                xtlk_op = gate.on(op_qubit, system_qubits[idx]) ** exponent
                xtlk_op_list.append(xtlk_op)
    return xtlk_op_list