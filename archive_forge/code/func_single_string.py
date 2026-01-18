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
def single_string(self):
    """Creates a long string with the ascii art.
        Returns:
            str: The lines joined by a newline (``\\n``)
        """
    if self._single_string:
        return self._single_string
    try:
        self._single_string = '\n'.join(self.lines()).encode(self.encoding).decode(self.encoding)
    except (UnicodeEncodeError, UnicodeDecodeError):
        warn('The encoding %s has a limited charset. Consider a different encoding in your environment. UTF-8 is being used instead' % self.encoding, RuntimeWarning)
        self.encoding = 'utf-8'
        self._single_string = '\n'.join(self.lines()).encode(self.encoding).decode(self.encoding)
    return self._single_string