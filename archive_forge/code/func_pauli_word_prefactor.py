from typing import Union
from functools import singledispatch
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.operation import Tensor
from .utils import is_pauli_word
from .conversion import pauli_sentence
def pauli_word_prefactor(observable):
    """If the operator provided is a valid Pauli word (i.e a single term which may be a tensor product
    of pauli operators), then this function extracts the prefactor.

    Args:
        observable (~.Operator): the operator to be examined

    Returns:
        Union[int, float, complex]: The scaling/phase coefficient of the Pauli word.

    Raises:
        ValueError: If an operator is provided that is not a valid Pauli word.

    **Example**

    >>> pauli_word_prefactor(qml.Identity(0))
    1
    >>> pauli_word_prefactor(qml.X(0) @ qml.Y(1))
    1
    >>> pauli_word_prefactor(qml.X(0) @ qml.Y(0))
    1j
    """
    return _pauli_word_prefactor(observable)