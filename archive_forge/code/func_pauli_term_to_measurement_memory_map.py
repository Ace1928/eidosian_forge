import itertools
from typing import Dict, List, Tuple, cast
import numpy as np
from pyquil.paulis import PauliTerm
def pauli_term_to_measurement_memory_map(term: PauliTerm, label: str='measurement') -> Dict[str, List[float]]:
    """
    Given a ``PauliTerm``, create a memory map corresponding to the ZXZXZ-decomposed single-qubit
    gates that allow for measurement in the eigenbasis of the ``PauliTerm``. For example, if we
    have the following program:

        RZ(measurement_alpha[0]) 0
        RX(pi/2) 0
        RZ(measurement_beta[0]) 0
        RX(-pi/2) 0
        RZ(measurement_gamma[0]) 0
        MEASURE 0 ro[0]

    We can measure in the ``Y`` basis (by default we measure in the ``Z`` basis) by providing the
    following memory map (which corresponds to ``RX(pi/2)``):

        {'measurement_alpha': [pi/2], 'measurement_beta': [pi/2], 'measurement_gamma': [pi/2]}

    :param term: The ``PauliTerm`` in question.
    :param label: The prefix to provide to ``pauli_term_to_euler_memory_map``, for labeling the
        declared memory regions. Defaults to "measurement".
    :return: Memory map for measuring in the desired basis.
    """
    return pauli_term_to_euler_memory_map(term, prefix=label, tuple_x=M_X, tuple_y=M_Y, tuple_z=M_Z)