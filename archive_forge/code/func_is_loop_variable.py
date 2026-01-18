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
def is_loop_variable(circuit, parameter):
    """Recurse into the instructions a parameter is used in, checking at every level if it is
        used as the loop variable of a ``for`` loop."""
    for instruction, index in circuit._parameter_table[parameter]:
        if isinstance(instruction, ForLoopOp):
            if index == 1:
                return True
        if isinstance(instruction, ControlFlowOp):
            if is_loop_variable(instruction.params[index], parameter):
                return True
    return False