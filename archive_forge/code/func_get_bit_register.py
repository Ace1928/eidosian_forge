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
def get_bit_register(circuit, bit):
    """Get the register for a bit if there is one

    Args:
        circuit (QuantumCircuit): the circuit being drawn
        bit (Qubit, Clbit): the bit to use to find the register and indexes

    Returns:
        ClassicalRegister: register associated with the bit
    """
    bit_loc = circuit.find_bit(bit)
    return bit_loc.registers[0][0] if bit_loc.registers else None