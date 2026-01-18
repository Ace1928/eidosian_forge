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
def should_compress(self, top_line, bot_line):
    """Decides if the top_line and bot_line should be merged,
        based on `self.vertical_compression`."""
    if self.vertical_compression == 'high':
        return True
    if self.vertical_compression == 'low':
        return False
    for top, bot in zip(top_line, bot_line):
        if top in ['┴', '╨'] and bot in ['┬', '╥']:
            return False
        if top.isalnum() and bot != ' ' or (bot.isalnum() and top != ' '):
            return False
    return True