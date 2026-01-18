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
def set_cl_multibox(self, condition, wire_map, top_connect='┴'):
    """Sets the multi clbit box.

        Args:
            condition (list[Union(Clbit, ClassicalRegister), int]): The condition
            wire_map (dict): Map of bits to indices
            top_connect (char): The char to connect the box on the top.

        Returns:
            List: list of tuples of connections between clbits for multi-bit conditions
        """
    if isinstance(condition, expr.Expr):
        label = '[expr]'
        out = []
        condition_bits = node_resources(condition).clbits
        registers = collections.defaultdict(list)
        for bit in condition_bits:
            registers[get_bit_register(self._circuit, bit)].append(bit)
        if (registerless := registers.pop(None, ())):
            out.extend(self.set_cond_bullets(label, ['1'] * len(registerless), registerless, wire_map))
        if self.cregbundle:
            for register in registers:
                self.set_clbit(register[0], BoxOnClWire(label=label, top_connect=top_connect))
        else:
            for register, bits in registers.items():
                out.extend(self.set_cond_bullets(label, ['1'] * len(bits), bits, wire_map))
        return out
    label, val_bits = get_condition_label_val(condition, self._circuit, self.cregbundle)
    if isinstance(condition[0], ClassicalRegister):
        cond_reg = condition[0]
    else:
        cond_reg = get_bit_register(self._circuit, condition[0])
    if self.cregbundle:
        if isinstance(condition[0], Clbit):
            if cond_reg is None:
                self.set_cond_bullets(label, val_bits, [condition[0]], wire_map)
            else:
                self.set_clbit(condition[0], BoxOnClWire(label=label, top_connect=top_connect))
        else:
            self.set_clbit(condition[0][0], BoxOnClWire(label=label, top_connect=top_connect))
        return []
    else:
        if isinstance(condition[0], Clbit):
            clbits = [condition[0]]
        else:
            clbits = [cond_reg[idx] for idx in range(cond_reg.size)]
        return self.set_cond_bullets(label, val_bits, clbits, wire_map)