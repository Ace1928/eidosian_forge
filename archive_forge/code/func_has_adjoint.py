import warnings
import itertools
from copy import copy
from typing import List
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.qubit import Hamiltonian
from pennylane.queuing import QueuingManager
from .composite import CompositeOp
@property
def has_adjoint(self):
    return True