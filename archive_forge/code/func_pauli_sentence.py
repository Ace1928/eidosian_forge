from functools import reduce, singledispatch
from itertools import product
from operator import matmul
from typing import Union, Tuple
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform
from .pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z, op_map
from .utils import is_pauli_word
def pauli_sentence(op):
    """Return the PauliSentence representation of an arithmetic operator or Hamiltonian.

    Args:
        op (~.Operator): The operator or Hamiltonian that needs to be converted.

    Raises:
        ValueError: Op must be a linear combination of Pauli operators

    Returns:
        .PauliSentence: the PauliSentence representation of an arithmetic operator or Hamiltonian
    """
    if isinstance(op, PauliWord):
        return PauliSentence({op: 1.0})
    if isinstance(op, PauliSentence):
        return op
    if (ps := op.pauli_rep) is not None:
        return ps
    return _pauli_sentence(op)