import warnings
from copy import copy
from functools import reduce, lru_cache
from typing import Iterable
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
@property
def pauli_rep(self):
    """Trivial pauli_rep"""
    return self