import itertools
from typing import Dict, List, Tuple, cast
import numpy as np
from pyquil.paulis import PauliTerm
def pauli_term_to_preparation_memory_map(term: PauliTerm, label: str='preparation') -> Dict[str, List[float]]:
    """
    Given a ``PauliTerm``, create a memory map corresponding to the ZXZXZ-decomposed single-qubit
    gates that prepare the plus one eigenstate of the ``PauliTerm``. For example, if we have the
    following program:

        RZ(preparation_alpha[0]) 0
        RX(pi/2) 0
        RZ(preparation_beta[0]) 0
        RX(-pi/2) 0
        RZ(preparation_gamma[0]) 0

    We can prepare the ``|+>`` state (by default we start in the ``|0>`` state) by providing the
    following memory map (which corresponds to ``RY(pi/2)``):

        {'preparation_alpha': [0.0], 'preparation_beta': [pi/2], 'preparation_gamma': [0.0]}

    :param term: The ``PauliTerm`` in question.
    :param label: The prefix to provide to ``pauli_term_to_euler_memory_map``, for labeling the
        declared memory regions. Defaults to "preparation".
    :return: Memory map for preparing the desired state.
    """
    return pauli_term_to_euler_memory_map(term, prefix=label, tuple_x=P_X, tuple_y=P_Y, tuple_z=P_Z)