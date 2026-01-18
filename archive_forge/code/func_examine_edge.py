import time
import logging
from functools import singledispatchmethod
from itertools import zip_longest
from collections import defaultdict
import rustworkx
from qiskit.circuit import (
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.equivalence import Key, NodeData
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
def examine_edge(self, edge):
    _, target, edata = edge
    if edata is None:
        return
    self._num_gates_remain_for_rule[edata.index] -= 1
    target = self.graph[target].key
    if self._num_gates_remain_for_rule[edata.index] > 0 or target in self.target_basis:
        raise rustworkx.visit.PruneSearch