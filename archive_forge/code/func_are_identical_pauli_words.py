from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
def are_identical_pauli_words(pauli_1, pauli_2):
    """Performs a check if two Pauli words have the same ``wires`` and ``name`` attributes.

    This is a convenience function that checks if two given :class:`~.Tensor` or :class:`~.Prod`
    instances specify the same Pauli word.

    Args:
        pauli_1 (Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]): the first Pauli word
        pauli_2 (Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]): the second Pauli word

    Returns:
        bool: whether ``pauli_1`` and ``pauli_2`` have the same wires and name attributes

    Raises:
        TypeError: if ``pauli_1`` or ``pauli_2`` are not :class:`~.Identity`, :class:`~.PauliX`,
            :class:`~.PauliY`, :class:`~.PauliZ`, :class:`~.Tensor`, :class:`~.SProd`, or
            :class:`~.Prod` instances

    **Example**

    >>> are_identical_pauli_words(qml.Z(0) @ qml.Z(1), qml.Z(0) @ qml.Z(1))
    True
    >>> are_identical_pauli_words(qml.Z(0) @ qml.Z(1), qml.Z(0) @ qml.X(3))
    False
    """
    if not (is_pauli_word(pauli_1) and is_pauli_word(pauli_2)):
        raise TypeError(f'Expected Pauli word observables, instead got {pauli_1} and {pauli_2}.')
    if pauli_1.pauli_rep is not None and pauli_2.pauli_rep is not None:
        return next(iter(pauli_1.pauli_rep)) == next(iter(pauli_2.pauli_rep))
    return False