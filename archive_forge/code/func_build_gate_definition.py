import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union
from qiskit.circuit import (
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check
from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter
def build_gate_definition(self, gate):
    """Builds a QuantumGateDefinition"""
    if isinstance(gate, standard_gates.CXGate):
        control, target = (ast.Identifier('c'), ast.Identifier('t'))
        call = ast.QuantumGateCall(ast.Identifier('U'), [control, target], parameters=[ast.Constant.PI, ast.IntegerLiteral(0), ast.Constant.PI], modifiers=[ast.QuantumGateModifier(ast.QuantumGateModifierName.CTRL)])
        return ast.QuantumGateDefinition(ast.QuantumGateSignature(ast.Identifier('cx'), [control, target]), ast.QuantumBlock([call]))
    self.push_context(gate.definition)
    signature = self.build_gate_signature(gate)
    body = ast.QuantumBlock(self.build_quantum_instructions(gate.definition.data))
    self.pop_context()
    return ast.QuantumGateDefinition(signature, body)