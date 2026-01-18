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
def slide_from_right(self, node, index):
    """Insert node into rightmost layer as long there is no conflict."""
    if not self:
        self.insert(0, [node])
        inserted = True
    else:
        inserted = False
        curr_index = index
        last_insertable_index = None
        while curr_index < len(self):
            if self.is_found_in(node, self[curr_index]):
                break
            if self.insertable(node, self[curr_index]):
                last_insertable_index = curr_index
            curr_index = curr_index + 1
        if last_insertable_index:
            self[last_insertable_index].append(node)
            inserted = True
        else:
            curr_index = index
            while curr_index > -1:
                if self.insertable(node, self[curr_index]):
                    self[curr_index].append(node)
                    inserted = True
                    break
                curr_index = curr_index - 1
    if not inserted:
        self.insert(0, [node])