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
def split_barriers(dag: DAGCircuit):
    """Mutate an input dag to split barriers into single qubit barriers."""
    for node in dag.op_nodes(Barrier):
        num_qubits = len(node.qargs)
        if num_qubits == 1:
            continue
        if node.op.label:
            barrier_uuid = f'{node.op.label}_uuid={uuid.uuid4()}'
        else:
            barrier_uuid = f'_none_uuid={uuid.uuid4()}'
        split_dag = DAGCircuit()
        split_dag.add_qubits([Qubit() for _ in range(num_qubits)])
        for i in range(num_qubits):
            split_dag.apply_operation_back(Barrier(1, label=barrier_uuid), qargs=(split_dag.qubits[i],), check=False)
        dag.substitute_node_with_dag(node, split_dag)