import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def pauli_kraus_map(probabilities: Sequence[float]) -> List[np.ndarray]:
    """
    Generate the Kraus operators corresponding to a pauli channel.

    :params probabilities: The 4^num_qubits list of probabilities specifying the
        desired pauli channel. There should be either 4 or 16 probabilities specified in the
        order I, X, Y, Z for 1 qubit or II, IX, IY, IZ, XI, XX, XY, etc for 2 qubits.

            For example::

                The d-dimensional depolarizing channel \\Delta parameterized as
                \\Delta(\\rho) = p \\rho + [(1-p)/d] I
                is specified by the list of probabilities
                [p + (1-p)/d, (1-p)/d,  (1-p)/d), ... , (1-p)/d)]

    :return: A list of the 4^num_qubits Kraus operators that parametrize the map.
    """
    if len(probabilities) not in [4, 16]:
        raise ValueError('Currently we only support one or two qubits, so the provided list of probabilities must have length 4 or 16.')
    if not np.allclose(sum(probabilities), 1.0, atol=0.001):
        raise ValueError('Probabilities must sum to one.')
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    if len(probabilities) == 4:
        operators = paulis
    else:
        operators = np.kron(paulis, paulis)
    return [coeff * op for coeff, op in zip(np.sqrt(probabilities), operators)]