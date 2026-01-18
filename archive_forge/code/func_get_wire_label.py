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
def get_wire_label(drawer, register, index, layout=None, cregbundle=True):
    """Get the bit labels to display to the left of the wires.

    Args:
        drawer (str): which drawer is calling ("text", "mpl", or "latex")
        register (QuantumRegister or ClassicalRegister): get wire_label for this register
        index (int): index of bit in register
        layout (Layout): Optional. mapping of virtual to physical bits
        cregbundle (bool): Optional. if set True bundle classical registers.
            Default: ``True``.

    Returns:
        str: label to display for the register/index
    """
    index_str = f'{index}' if drawer == 'text' else f'{{{index}}}'
    if register is None:
        wire_label = index_str
        return wire_label
    if drawer == 'text':
        reg_name = f'{register.name}'
        reg_name_index = f'{register.name}_{index}'
    else:
        reg_name = f'{{{fix_special_characters(register.name)}}}'
        reg_name_index = f'{reg_name}_{{{index}}}'
    if isinstance(register, ClassicalRegister):
        if cregbundle and drawer != 'latex':
            wire_label = f'{register.name}'
            return wire_label
        if register.size == 1 or cregbundle:
            wire_label = reg_name
        else:
            wire_label = reg_name_index
        return wire_label
    if register.size == 1:
        wire_label = reg_name
    elif layout is None:
        wire_label = reg_name_index
    elif layout[index]:
        virt_bit = layout[index]
        try:
            virt_reg = next((reg for reg in layout.get_registers() if virt_bit in reg))
            if drawer == 'text':
                wire_label = f'{virt_reg.name}_{virt_reg[:].index(virt_bit)} -> {index}'
            else:
                wire_label = f'{{{virt_reg.name}}}_{{{virt_reg[:].index(virt_bit)}}} \\mapsto {{{index}}}'
        except StopIteration:
            if drawer == 'text':
                wire_label = f'{virt_bit} -> {index}'
            else:
                wire_label = f'{{{virt_bit}}} \\mapsto {{{index}}}'
        if drawer != 'text':
            wire_label = wire_label.replace(' ', '\\;')
    else:
        wire_label = index_str
    return wire_label