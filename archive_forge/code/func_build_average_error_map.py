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
def build_average_error_map(target, properties, coupling_map):
    """Build an average error map used for scoring layouts pre-basis translation."""
    num_qubits = 0
    if target is not None:
        num_qubits = target.num_qubits
        avg_map = ErrorMap(len(target.qargs))
    elif coupling_map is not None:
        num_qubits = coupling_map.size()
        avg_map = ErrorMap(num_qubits + coupling_map.graph.num_edges())
    else:
        avg_map = ErrorMap(0)
    built = False
    if target is not None:
        for qargs in target.qargs:
            if qargs is None:
                continue
            qarg_error = 0.0
            count = 0
            for op in target.operation_names_for_qargs(qargs):
                inst_props = target[op].get(qargs, None)
                if inst_props is not None and inst_props.error is not None:
                    count += 1
                    qarg_error += inst_props.error
            if count > 0:
                if len(qargs) == 1:
                    qargs = (qargs[0], qargs[0])
                avg_map.add_error(qargs, qarg_error / count)
                built = True
    elif properties is not None:
        errors = defaultdict(list)
        for qubit in range(len(properties.qubits)):
            errors[qubit,].append(properties.readout_error(qubit))
        for gate in properties.gates:
            qubits = tuple(gate.qubits)
            for param in gate.parameters:
                if param.name == 'gate_error':
                    errors[qubits].append(param.value)
        for k, v in errors.items():
            if len(k) == 1:
                qargs = (k[0], k[0])
            else:
                qargs = k
            if qargs[0] >= num_qubits or qargs[1] >= num_qubits:
                continue
            avg_map.add_error(qargs, statistics.mean(v))
            built = True
    if not built and target is not None and (coupling_map is None):
        coupling_map = target.build_coupling_map()
    if not built and coupling_map is not None:
        for qubit in range(num_qubits):
            avg_map.add_error((qubit, qubit), (coupling_map.graph.out_degree(qubit) + coupling_map.graph.in_degree(qubit)) / num_qubits)
        for edge in coupling_map.graph.edge_list():
            avg_map.add_error(edge, (avg_map[edge[0], edge[0]] + avg_map[edge[1], edge[1]]) / 2)
            built = True
    if built:
        return avg_map
    else:
        return None