from collections import defaultdict
import statistics
import random
import numpy as np
from rustworkx import PyDiGraph, PyGraph, connected_components
from qiskit.circuit import ControlFlowOp, ForLoopOp
from qiskit.converters import circuit_to_dag
from qiskit._accelerate import vf2_layout
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.error_map import ErrorMap
def map_free_qubits(free_nodes, partial_layout, num_physical_qubits, reverse_bit_map, avg_error_map):
    """Add any free nodes to a layout."""
    if not free_nodes:
        return partial_layout
    if avg_error_map is not None:
        free_qubits = sorted(set(range(num_physical_qubits)) - partial_layout.get_physical_bits().keys(), key=lambda bit: avg_error_map.get((bit, bit), 1.0))
    else:
        free_qubits = list(set(range(num_physical_qubits)) - partial_layout.get_physical_bits().keys())
    for im_index in sorted(free_nodes, key=lambda x: sum(free_nodes[x].values())):
        if not free_qubits:
            return None
        selected_qubit = free_qubits.pop(0)
        partial_layout.add(reverse_bit_map[im_index], selected_qubit)
    return partial_layout