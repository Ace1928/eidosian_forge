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
def get_gate_ctrl_text(op, drawer, style=None, calibrations=None):
    """Load the gate_text and ctrl_text strings based on names and labels"""
    anno_list = []
    anno_text = ''
    if isinstance(op, AnnotatedOperation) and op.modifiers:
        for modifier in op.modifiers:
            if isinstance(modifier, InverseModifier):
                anno_list.append('Inv')
            elif isinstance(modifier, PowerModifier):
                anno_list.append('Pow(' + str(round(modifier.power, 1)) + ')')
        anno_text = ', '.join(anno_list)
    op_label = getattr(op, 'label', None)
    op_type = type(op)
    base_name = base_label = base_type = None
    if hasattr(op, 'base_gate'):
        base_name = op.base_gate.name
        base_label = op.base_gate.label
        base_type = type(op.base_gate)
    if hasattr(op, 'base_op'):
        base_name = op.base_op.name
    ctrl_text = None
    if base_label:
        gate_text = base_label
        ctrl_text = op_label
    elif op_label and isinstance(op, ControlledGate):
        gate_text = base_name
        ctrl_text = op_label
    elif op_label:
        gate_text = op_label
    elif base_name:
        gate_text = base_name
    else:
        gate_text = op.name
    raw_gate_text = op.name if gate_text == base_name else gate_text
    if drawer != 'text' and gate_text in style['disptex']:
        if style['disptex'][gate_text][0] == '$' and style['disptex'][gate_text][-1] == '$':
            gate_text = style['disptex'][gate_text]
        else:
            gate_text = f'$\\mathrm{{{style['disptex'][gate_text]}}}$'
    elif drawer == 'latex':
        if _is_boolean_expression(gate_text, op):
            gate_text = gate_text.replace('~', '$\\neg$').replace('&', '\\&')
            gate_text = f'$\\texttt{{{gate_text}}}$'
        elif (gate_text == op.name and op_type not in (Gate, Instruction) or (gate_text == base_name and base_type not in (Gate, Instruction))) and op_type is not PauliEvolutionGate:
            gate_text = f'$\\mathrm{{{gate_text.capitalize()}}}$'
        else:
            gate_text = f'$\\mathrm{{{gate_text}}}$'
            gate_text = gate_text.replace('_', '\\_')
            gate_text = gate_text.replace('^', '\\string^')
            gate_text = gate_text.replace('-', '\\mbox{-}')
        ctrl_text = f'$\\mathrm{{{ctrl_text}}}$'
    elif (gate_text == op.name and op_type not in (Gate, Instruction) or (gate_text == base_name and base_type not in (Gate, Instruction))) and op_type is not PauliEvolutionGate:
        gate_text = gate_text.capitalize()
    if drawer == 'mpl' and op.name in calibrations:
        if isinstance(op, ControlledGate):
            ctrl_text = '' if ctrl_text is None else ctrl_text
            ctrl_text = '(cal)\n' + ctrl_text
        else:
            gate_text = gate_text + '\n(cal)'
    if anno_text:
        gate_text += ' - ' + anno_text
    return (gate_text, ctrl_text, raw_gate_text)