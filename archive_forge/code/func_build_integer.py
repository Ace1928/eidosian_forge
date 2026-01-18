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
def build_integer(self, value) -> ast.IntegerLiteral:
    """Build an integer literal, raising a :obj:`.QASM3ExporterError` if the input is not
        actually an
        integer."""
    if not isinstance(value, numbers.Integral):
        raise QASM3ExporterError(f"'{value}' is not an integer")
    return ast.IntegerLiteral(int(value))