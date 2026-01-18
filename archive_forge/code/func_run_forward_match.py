import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def run_forward_match(self):
    """Apply the forward match algorithm and returns the list of matches given an initial match
        and a qubits configuration.
        """
    self._init_successors_to_visit()
    self._init_matched_with_circuit()
    self._init_matched_with_pattern()
    self._init_is_blocked_circuit()
    self._init_list_match()
    self._init_matched_nodes()
    while self.matched_nodes_list:
        v_first, successors_to_visit = self._get_node_forward(0)
        self._remove_node_forward(0)
        if not successors_to_visit:
            continue
        label = successors_to_visit[0]
        v = [label, self.circuit_dag.get_node(label)]
        successors_to_visit.pop(0)
        self.matched_nodes_list.append([v_first.node_id, v_first, successors_to_visit])
        self.matched_nodes_list.sort(key=lambda x: x[2])
        if self.circuit_blocked[v[0]] | (self.circuit_matched_with[v[0]] != []):
            continue
        self._find_forward_candidates(self.circuit_matched_with[v_first.node_id][0])
        match = False
        for i in self.candidates:
            if match:
                break
            node_circuit = self.circuit_dag.get_node(label)
            node_pattern = self.pattern_dag.get_node(i)
            if len(self.wires[label]) != len(node_pattern.wires) or set(self.wires[label]) != set(node_pattern.wires) or node_circuit.op.name != node_pattern.op.name:
                continue
            if _compare_operation_without_qubits(node_circuit, node_pattern):
                if _compare_qubits(node_circuit, self.wires[label], self.target_wires[label], self.control_wires[label], node_pattern.wires, node_pattern.control_wires, node_pattern.target_wires):
                    self.circuit_matched_with[label] = [i]
                    self.pattern_matched_with[i] = [label]
                    self.match.append([i, label])
                    potential = self.circuit_dag.direct_successors(label)
                    for potential_id in potential:
                        if self.circuit_blocked[potential_id] | (self.circuit_matched_with[potential_id] != []):
                            potential.remove(potential_id)
                    sorted_potential = sorted(potential)
                    successorstovisit = sorted_potential
                    self.matched_nodes_list.append([v[0], v[1], successorstovisit])
                    self.matched_nodes_list.sort(key=lambda x: x[2])
                    match = True
                    continue
        if not match:
            self.circuit_blocked[label] = True
            for succ in v[1].successors:
                self.circuit_blocked[succ] = True