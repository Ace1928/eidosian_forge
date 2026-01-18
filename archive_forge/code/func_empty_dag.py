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
def empty_dag(block):
    empty = DAGCircuit()
    empty.add_qubits(out_dag.qubits)
    for qreg in out_dag.qregs.values():
        empty.add_qreg(qreg)
    empty.add_clbits(block.clbits)
    for creg in block.cregs:
        empty.add_creg(creg)
    empty._global_phase = block.global_phase
    return empty