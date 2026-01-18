import re
from collections import OrderedDict
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ClassicalRegister, QuantumCircuit, Qubit, ControlFlowOp
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier, PowerModifier
from qiskit.circuit.tools import pi_check
from qiskit.converters import circuit_to_dag
from qiskit.utils import optionals as _optionals
from ..exceptions import VisualizationError
def is_found_in(self, node, nodes):
    """Is any qreq in node found in any of nodes?"""
    all_qargs = []
    for a_node in nodes:
        for qarg in a_node.qargs:
            all_qargs.append(qarg)
    return any((i in node.qargs for i in all_qargs))