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
def build_bit_list(im_graph, bit_map):
    """Generate a bit list for scoring."""
    bit_list = np.zeros(len(im_graph), dtype=np.int32)
    for node_index in bit_map.values():
        try:
            bit_list[node_index] = sum(im_graph[node_index].values())
        except IndexError:
            pass
    return bit_list