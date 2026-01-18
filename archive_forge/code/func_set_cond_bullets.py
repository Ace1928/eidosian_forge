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
def set_cond_bullets(self, label, val_bits, clbits, wire_map):
    """Sets bullets for classical conditioning when cregbundle=False.

        Args:
            label (str): String to display below the condition
            val_bits (list(int)): A list of bit values
            clbits (list[Clbit]): The list of classical bits on
                which the instruction is conditioned.
            wire_map (dict): Map of bits to indices

        Returns:
            List: list of tuples of open or closed bullets for condition bits
        """
    current_cons = []
    wire_max = max((wire_map[bit] for bit in clbits))
    for i, bit in enumerate(clbits):
        bot_connect = ' '
        if wire_map[bit] == wire_max:
            bot_connect = label
        if val_bits[i] == '1':
            self.clbit_layer[wire_map[bit] - len(self.qubits)] = ClBullet(top_connect='║', bot_connect=bot_connect)
        elif val_bits[i] == '0':
            self.clbit_layer[wire_map[bit] - len(self.qubits)] = ClOpenBullet(top_connect='║', bot_connect=bot_connect)
        actual_index = wire_map[bit]
        if actual_index not in [i for i, j in current_cons]:
            current_cons.append((actual_index, self.clbit_layer[wire_map[bit] - len(self.qubits)]))
    return current_cons