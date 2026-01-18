from __future__ import annotations
from collections import defaultdict
from typing import List, Callable, TypeVar, Dict, Union
import uuid
import rustworkx as rx
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit import Qubit, Barrier, Clbit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOutNode
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils
def map_components(dag_components: List[DAGCircuit], cmap_components: List[CouplingMap]) -> Dict[int, List[int]]:
    """Returns a map where the key is the index of each connected component in cmap_components and
    the value is a list of indices from dag_components which should be placed onto it."""
    free_qubits = {index: len(cmap.graph) for index, cmap in enumerate(cmap_components)}
    out_mapping = defaultdict(list)
    for dag_index, dag in sorted(enumerate(dag_components), key=lambda x: x[1].num_qubits(), reverse=True):
        for cmap_index in sorted(range(len(cmap_components)), key=lambda index: free_qubits[index], reverse=True):
            if dag.num_qubits() <= free_qubits[cmap_index]:
                out_mapping[cmap_index].append(dag_index)
                free_qubits[cmap_index] -= dag.num_qubits()
                break
        else:
            raise TranspilerError('A connected component of the DAGCircuit is too large for any of the connected components in the coupling map.')
    return out_mapping