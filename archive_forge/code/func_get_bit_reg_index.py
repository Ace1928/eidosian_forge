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
def get_bit_reg_index(circuit, bit):
    """Get the register for a bit if there is one, and the index of the bit
    from the top of the circuit, or the index of the bit within a register.

    Args:
        circuit (QuantumCircuit): the circuit being drawn
        bit (Qubit, Clbit): the bit to use to find the register and indexes

    Returns:
        (ClassicalRegister, None): register associated with the bit
        int: index of the bit from the top of the circuit
        int: index of the bit within the register, if there is a register
    """
    bit_loc = circuit.find_bit(bit)
    bit_index = bit_loc.index
    register, reg_index = bit_loc.registers[0] if bit_loc.registers else (None, None)
    return (register, bit_index, reg_index)