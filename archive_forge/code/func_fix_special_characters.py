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
def fix_special_characters(label):
    """
    Convert any special characters for mpl and latex drawers.
    Currently only checks for multiple underscores in register names
    and uses wider space for mpl and latex drawers.

    Args:
        label (str): the label to fix

    Returns:
        str: label to display
    """
    label = label.replace('_', '\\_').replace(' ', '\\;')
    return label