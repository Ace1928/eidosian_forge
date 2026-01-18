import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@property
def has_matrix(self):
    return all((op.has_matrix or isinstance(op, qml.Hamiltonian) for op in self))