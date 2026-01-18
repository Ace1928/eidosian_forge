from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def get_1d_Ising_lcu_coeffs(n_spins: int, j_zz_strength: float=np.pi / 3, gamma_x_strength: float=np.pi / 7) -> np.ndarray:
    """Get LCU coefficients for a 1d ising Hamiltonian.

    The order of the terms is according to `get_1d_Ising_hamiltonian`, namely: ZZ's and X's
    interleaved.
    """
    spins = cirq.LineQubit.range(n_spins)
    ham = get_1d_Ising_hamiltonian(spins, j_zz_strength, gamma_x_strength)
    coeffs = np.array([term.coefficient.real for term in ham])
    lcu_coeffs = coeffs / np.sum(coeffs)
    return lcu_coeffs