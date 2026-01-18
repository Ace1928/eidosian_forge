from functools import wraps
import pennylane as qml
from pennylane.math import conj, moveaxis, transpose
from pennylane.operation import Observable, Operation, Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import make_qscript
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
from .symbolicop import SymbolicOp
@property
def parameter_frequencies(self):
    return self.base.parameter_frequencies