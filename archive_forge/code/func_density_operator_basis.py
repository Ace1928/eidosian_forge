import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
def density_operator_basis(n_qubits: int) -> Iterator[np.ndarray]:
    """Yields operator basis consisting of density operators."""
    RHO_0 = np.array([[1, 0], [0, 0]], dtype=np.complex64)
    RHO_1 = np.array([[0, 0], [0, 1]], dtype=np.complex64)
    RHO_2 = np.array([[1, 1], [1, 1]], dtype=np.complex64) / 2
    RHO_3 = np.array([[1, -1j], [1j, 1]], dtype=np.complex64) / 2
    RHO_BASIS = (RHO_0, RHO_1, RHO_2, RHO_3)
    if n_qubits < 1:
        yield np.array(1)
        return
    for rho1 in RHO_BASIS:
        for rho2 in density_operator_basis(n_qubits - 1):
            yield np.kron(rho1, rho2)