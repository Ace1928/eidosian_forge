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
def discover_vertex(self, v, score):
    gate = self.graph[v].key
    self._source_gates_remain.discard(gate)
    self._opt_cost_map[gate] = score
    rule = self._predecessors.get(gate, None)
    if rule is not None:
        logger.debug('Gate %s generated using rule \n%s\n with total cost of %s.', gate.name, rule.circuit, score)
        self._basis_transforms.append((gate.name, gate.num_qubits, rule.params, rule.circuit))
    if not self._source_gates_remain:
        self._basis_transforms.reverse()
        raise StopIfBasisRewritable