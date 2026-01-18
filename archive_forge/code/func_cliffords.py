from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp
@property
def cliffords(self) -> list[SparsePauliOp]:
    """
        Get clifford operators, built based on symmetries and single-qubit X.

        Returns:
            A list of unitaries used to diagonalize the Hamiltonian.
        """
    cliffords = [(SparsePauliOp(pauli_symm) + SparsePauliOp(sq_pauli)) / np.sqrt(2) for pauli_symm, sq_pauli in zip(self._symmetries, self._sq_paulis)]
    return cliffords