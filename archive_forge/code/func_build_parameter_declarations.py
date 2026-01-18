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
def build_parameter_declarations(self):
    """Builds lists of the input, output and standard variables used in this program."""
    global_scope = self.global_scope(assert_=True)
    for parameter in global_scope.circuit.parameters:
        parameter_name = self._register_variable(parameter, global_scope)
        declaration = _infer_variable_declaration(global_scope.circuit, parameter, parameter_name)
        if declaration is None:
            continue
        if isinstance(declaration, ast.IODeclaration):
            self._global_io_declarations.append(declaration)
        else:
            self._global_classical_declarations.append(declaration)