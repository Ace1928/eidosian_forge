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
def string_to_pauli_word(pauli_string, wire_map=None):
    """Convert a string in terms of ``'I'``, ``'X'``, ``'Y'``, and ``'Z'`` into a Pauli word
    for the given wire map.

    Args:
        pauli_string (str): A string of characters consisting of ``'I'``, ``'X'``, ``'Y'``, and ``'Z'``
            indicating a Pauli word.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        .Observable: The Pauli word representing of ``pauli_string`` on the wires
        enumerated in the wire map.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> string_to_pauli_word('XIY', wire_map=wire_map)
    X('a') @ Y('c')
    """
    character_map = {'I': Identity, 'X': PauliX, 'Y': PauliY, 'Z': PauliZ}
    if not isinstance(pauli_string, str):
        raise TypeError(f'Input to string_to_pauli_word must be string, obtained {pauli_string}')
    if any((char not in character_map for char in pauli_string)):
        raise ValueError(f"Invalid characters encountered in string_to_pauli_word string {pauli_string}. Permitted characters are 'I', 'X', 'Y', and 'Z'")
    if wire_map is None:
        wire_map = {x: x for x in range(len(pauli_string))}
    if len(pauli_string) != len(wire_map):
        raise ValueError('Wire map and pauli_string must have the same length to convert from string to Pauli word.')
    if pauli_string == 'I' * len(wire_map):
        first_wire = list(wire_map)[0]
        return Identity(first_wire)
    pauli_word = None
    for wire_name, wire_idx in wire_map.items():
        pauli_char = pauli_string[wire_idx]
        if pauli_char == 'I':
            continue
        if pauli_word is not None:
            pauli_word = pauli_word @ character_map[pauli_char](wire_name)
        else:
            pauli_word = character_map[pauli_char](wire_name)
    return pauli_word