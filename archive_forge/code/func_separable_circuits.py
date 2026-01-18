from node A to node B means that the (qu)bit passes from the output of A
from collections import OrderedDict, defaultdict, deque, namedtuple
import copy
import math
from typing import Dict, Generator, Any, List
import numpy as np
import rustworkx as rx
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources, node_resources, CONTROL_FLOW_OP_NAMES
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit.bit import Bit
def separable_circuits(self, remove_idle_qubits=False) -> List['DAGCircuit']:
    """Decompose the circuit into sets of qubits with no gates connecting them.

        Args:
            remove_idle_qubits (bool): Flag denoting whether to remove idle qubits from
                the separated circuits. If ``False``, each output circuit will contain the
                same number of qubits as ``self``.

        Returns:
            List[DAGCircuit]: The circuits resulting from separating ``self`` into sets
                of disconnected qubits

        Each :class:`~.DAGCircuit` instance returned by this method will contain the same number of
        clbits as ``self``. The global phase information in ``self`` will not be maintained
        in the subcircuits returned by this method.
        """
    connected_components = rx.weakly_connected_components(self._multi_graph)
    disconnected_subgraphs = []
    for components in connected_components:
        disconnected_subgraphs.append(self._multi_graph.subgraph(list(components)))

    def _key(x):
        return x.sort_key
    decomposed_dags = []
    for subgraph in disconnected_subgraphs:
        new_dag = self.copy_empty_like()
        new_dag.global_phase = 0
        subgraph_is_classical = True
        for node in rx.lexicographical_topological_sort(subgraph, key=_key):
            if isinstance(node, DAGInNode):
                if isinstance(node.wire, Qubit):
                    subgraph_is_classical = False
            if not isinstance(node, DAGOpNode):
                continue
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
        if not subgraph_is_classical:
            decomposed_dags.append(new_dag)
    if remove_idle_qubits:
        for dag in decomposed_dags:
            dag.remove_qubits(*(bit for bit in dag.idle_wires() if isinstance(bit, Qubit)))
    return decomposed_dags