import logging
from copy import deepcopy
import time
import rustworkx
from qiskit.circuit import SwitchCaseOp, ControlFlowOp, Clbit, ClassicalRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.converters import dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.dagcircuit import DAGCircuit
from qiskit.utils.parallel import CPU_COUNT
from qiskit._accelerate.sabre_swap import (
from qiskit._accelerate.nlayout import NLayout
def apply_swaps(dest_dag, swaps, layout):
    for a, b in swaps:
        qubits = (physical_qubits[a], physical_qubits[b])
        layout.swap_physical(a, b)
        dest_dag.apply_operation_back(swap_singleton, qubits, (), check=False)