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
def score_layout(avg_error_map, layout_mapping, bit_map, _reverse_bit_map, im_graph, strict_direction=False, run_in_parallel=False, edge_list=None, bit_list=None):
    """Score a layout given an average error map."""
    if layout_mapping:
        size = max(max(layout_mapping), max(layout_mapping.values()))
    else:
        size = 0
    nlayout = NLayout(layout_mapping, size + 1, size + 1)
    if bit_list is None:
        bit_list = build_bit_list(im_graph, bit_map)
    if edge_list is None:
        edge_list = build_edge_list(im_graph)
    return vf2_layout.score_layout(bit_list, edge_list, avg_error_map, nlayout, strict_direction, run_in_parallel)