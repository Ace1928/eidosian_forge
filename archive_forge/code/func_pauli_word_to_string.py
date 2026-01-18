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
def pauli_word_to_string(pauli_word, wire_map=None):
    """Convert a Pauli word to a string.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Tensor` instance containing Pauli operators.

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a Pauli operator.

    * A :class:`.Hamiltonian` instance with only one term.

    Given a Pauli in observable form, convert it into string of
    characters from ``['I', 'X', 'Y', 'Z']``. This representation is required for
    functions such as :class:`.PauliRot`.

    .. warning::

        This method ignores any potential coefficient multiplying the Pauli word:

        >>> qml.pauli.pauli_word_to_string(3 * qml.X(0) @ qml.Y(1))
        'XY'

    .. warning::

        This method assumes all Pauli operators are acting on different wires, ignoring
        any extra operators:

        >>> qml.pauli.pauli_word_to_string(qml.X(0) @ qml.Y(0) @ qml.Y(0))
        'X'

    Args:
        pauli_word (Observable): an observable, either a :class:`~.Tensor` instance or
            single-qubit observable representing a Pauli group element.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        str: The string representation of the observable in terms of ``'I'``, ``'X'``, ``'Y'``,
        and/or ``'Z'``.

    Raises:
        TypeError: if the input observable is not a proper Pauli word.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> pauli_word = qml.X('a') @ qml.Y('c')
    >>> pauli_word_to_string(pauli_word, wire_map=wire_map)
    'XIY'
    """
    if not is_pauli_word(pauli_word):
        raise TypeError(f'Expected Pauli word observables, instead got {pauli_word}')
    if isinstance(pauli_word, Hamiltonian):
        pauli_word = pauli_word.ops[0]
    elif isinstance(pauli_word, SProd):
        pauli_word = pauli_word.base
    if isinstance(pauli_word, Prod):
        pauli_word = Tensor(*pauli_word.operands)
    character_map = {'Identity': 'I', 'PauliX': 'X', 'PauliY': 'Y', 'PauliZ': 'Z'}
    if wire_map is None:
        wire_map = {pauli_word.wires.labels[i]: i for i in range(len(pauli_word.wires))}
    n_qubits = len(wire_map)
    pauli_string = ['I'] * n_qubits
    if not isinstance(pauli_word.name, list):
        if pauli_word.name != 'Identity':
            wire_idx = wire_map[pauli_word.wires[0]]
            pauli_string[wire_idx] = character_map[pauli_word.name]
        return ''.join(pauli_string)
    for name, wire_label in zip(pauli_word.name, pauli_word.wires):
        wire_idx = wire_map[wire_label]
        pauli_string[wire_idx] = character_map[name]
    return ''.join(pauli_string)