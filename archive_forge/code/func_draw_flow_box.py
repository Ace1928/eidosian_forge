from io import StringIO
from warnings import warn
from shutil import get_terminal_size
import collections
import sys
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit import ControlledGate, Reset, Measure
from qiskit.circuit import ControlFlowOp, WhileLoopOp, IfElseOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.classical import expr
from qiskit.circuit.controlflow import node_resources
from qiskit.circuit.library.standard_gates import IGate, RZZGate, SwapGate, SXGate, SXdgGate
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from ._utils import (
from ..exceptions import VisualizationError
def draw_flow_box(self, node, flow_wire_map, section, circ_num=0):
    """Draw the left, middle, or right of a control flow box"""
    op = node.op
    conditional = section == CF_LEFT and (not isinstance(op, ForLoopOp))
    depth = str(self._nest_depth)
    if section == CF_LEFT:
        etext = ''
        if self._expr_text:
            etext = ' ' + self._expr_text
        if isinstance(op, IfElseOp):
            label = 'If-' + depth + etext
        elif isinstance(op, WhileLoopOp):
            label = 'While-' + depth + etext
        elif isinstance(op, ForLoopOp):
            indexset = op.params[0]
            if 'range' not in str(indexset) and len(indexset) > 4:
                index_str = str(indexset[:4])
                index_str = index_str[:-1] + ', ...)'
            else:
                index_str = str(indexset)
            label = 'For-' + depth + ' ' + index_str
        else:
            label = 'Switch-' + depth + etext
    elif section == CF_MID:
        if isinstance(op, IfElseOp):
            label = 'Else-' + depth
        else:
            jump_list = []
            for jump_values, _ in list(op.cases_specifier()):
                jump_list.append(jump_values)
            if 'default' in str(jump_list[circ_num][0]):
                jump_str = 'default'
            else:
                jump_str = str(jump_list[circ_num]).replace(',)', ')')
            label = 'Case-' + depth + ' ' + jump_str
    else:
        label = 'End-' + depth
    flow_layer = Layer(self.qubits, self.clbits, self.cregbundle, self._circuit, flow_wire_map)
    if len(node.qargs) == 1:
        flow_layer.set_qubit(self.qubits[flow_wire_map[node.qargs[0]]], FlowOnQuWire(section, label=label, conditional=conditional))
    else:
        idx_list = [flow_wire_map[qarg] for qarg in node.qargs]
        min_idx = min(idx_list)
        max_idx = max(idx_list)
        box_height = max_idx - min_idx + 1
        flow_layer.set_qubit(self.qubits[min_idx], FlowOnQuWireTop(section, label=label, wire_label=''))
        for order, i in enumerate(range(min_idx + 1, max_idx)):
            flow_layer.set_qubit(self.qubits[i], FlowOnQuWireMid(section, label=label, input_length=box_height, order=order, wire_label=''))
        flow_layer.set_qubit(self.qubits[max_idx], FlowOnQuWireBot(section, label=label, input_length=box_height, conditional=conditional, wire_label=''))
    if conditional:
        if isinstance(node.op, SwitchCaseOp):
            if isinstance(node.op.target, expr.Expr):
                condition = node.op.target
            elif isinstance(node.op.target, Clbit):
                condition = (node.op.target, 1)
            else:
                condition = (node.op.target, 2 ** node.op.target.size - 1)
        else:
            condition = node.op.condition
        _ = flow_layer.set_cl_multibox(condition, flow_wire_map, top_connect='╨')
    return flow_layer