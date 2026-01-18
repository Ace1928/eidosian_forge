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
def draw_wires(self, wires):
    """Given a list of wires, creates a list of lines with the text drawing.

        Args:
            wires (list): A list of wires with nodes.
        Returns:
            list: A list of lines with the text drawing.
        """
    lines = []
    bot_line = None
    for wire in wires:
        top_line = ''
        for node in wire:
            top_line += node.top
        if bot_line is None:
            lines.append(top_line)
        elif self.should_compress(top_line, bot_line):
            lines.append(TextDrawing.merge_lines(lines.pop(), top_line))
        else:
            lines.append(TextDrawing.merge_lines(lines[-1], top_line, icod='bot'))
        mid_line = ''
        for node in wire:
            mid_line += node.mid
        lines.append(TextDrawing.merge_lines(lines[-1], mid_line, icod='bot'))
        bot_line = ''
        for node in wire:
            bot_line += node.bot
        lines.append(TextDrawing.merge_lines(lines[-1], bot_line, icod='bot'))
    return lines