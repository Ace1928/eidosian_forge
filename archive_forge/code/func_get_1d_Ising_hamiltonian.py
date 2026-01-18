from typing import List, Sequence
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def get_1d_Ising_hamiltonian(qubits: Sequence[cirq.Qid], j_zz_strength: float=1.0, gamma_x_strength: float=-1) -> cirq.PauliSum:
    """A one dimensional ising model with periodic boundaries.

    $$
    H = -J\\sum_{k=0}^{L-1}\\sigma_{k}^{Z}\\sigma_{(k+1)\\%L}^{Z} - \\Gamma\\sum_{k=0}^{L-1}\\sigma_{k}^{X}
    $$

    Args:
        qubits: One qubit for each spin site.
        j_zz_strength: The two-body ZZ potential strength, $J$.
        gamma_x_strength: The one-body X potential strength, $\\Gamma$.

    Returns:
        cirq.PauliSum representing the Hamiltonian
    """
    n_sites = len(qubits)
    terms: List[cirq.PauliString] = []
    for k in range(n_sites):
        terms.append(cirq.PauliString({qubits[k]: cirq.Z, qubits[(k + 1) % n_sites]: cirq.Z}, coefficient=j_zz_strength))
        terms.append(cirq.PauliString({qubits[k]: cirq.X}, coefficient=gamma_x_strength))
    return cirq.PauliSum.from_pauli_strings(terms)