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
def connect_with(self, wire_char):
    """Connects the elements in the layer using wire_char.

        Args:
            wire_char (char): For example '║' or '│'.
        """
    for label, affected_bits in self.connections:
        if not affected_bits:
            continue
        for index, affected_bit in enumerate(affected_bits):
            if isinstance(affected_bit, (ClBullet, ClOpenBullet)):
                wire_char = '║'
                if index == 0 and len(affected_bits) > 1:
                    affected_bit.connect(wire_char, ['bot'])
                elif index == len(affected_bits) - 1:
                    affected_bit.connect(wire_char, ['top'])
                else:
                    affected_bit.connect(wire_char, ['bot', 'top'])
            elif index == 0:
                affected_bit.connect(wire_char, ['bot'])
            elif index == len(affected_bits) - 1:
                affected_bit.connect(wire_char, ['top'], label)
            else:
                affected_bit.connect(wire_char, ['bot', 'top'])
        if label:
            for affected_bit in affected_bits:
                affected_bit.right_fill = len(label) + len(affected_bit.mid)
                if isinstance(affected_bit, (Bullet, OpenBullet)) and affected_bit.conditional:
                    affected_bit.left_fill = len(label) + len(affected_bit.mid)