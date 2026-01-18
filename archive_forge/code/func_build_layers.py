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
def build_layers(self):
    """
        Constructs layers.
        Returns:
            list: List of DrawElements.
        Raises:
            VisualizationError: When the drawing is, for some reason, impossible to be drawn.
        """
    wire_names = self.wire_names(with_initial_state=self.initial_state)
    if not wire_names:
        return []
    layers = [InputWire.fillup_layer(wire_names)]
    for node_layer in self.nodes:
        layer = Layer(self.qubits, self.clbits, self.cregbundle, self._circuit, self._wire_map)
        for node in node_layer:
            if isinstance(node.op, ControlFlowOp):
                self._nest_depth = 0
                self.add_control_flow(node, layers, self._wire_map)
            else:
                layer, current_cons, current_cons_cond, connection_label = self._node_to_gate(node, layer, self._wire_map)
                layer.connections.append((connection_label, current_cons))
                layer.connections.append((None, current_cons_cond))
        layer.connect_with('│')
        layers.append(layer.full_layer)
    return layers