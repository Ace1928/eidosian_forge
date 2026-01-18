import json
from os import path
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
def calculate_xi_decomposition(hamiltonian):
    """
    Calculates the Xi-decomposition from the given Hamiltonian by constructing the sparse matrix
    representing the Hamiltonian, finding its spectrum and then construct projectors and
    eigenvalue spacings

    Definition of the Xi decomposition of operator O:

    .. math::
        \\frac{\\lambda_0 +\\lambda_J}{2} \\mathbb{1} + \\sum_{x=1}^{J-1} \\frac{\\delta \\lambda_x}{2}\\Xi_x ,

    where the lambdas are the sorted eigenvalues of O and

    ..math::
       \\Xi_x = \\mathbb{1} - \\sum_(j<x) 2 \\Pi_j \\,, \\quad \\delta \\lambda_x = \\lambda_x - \\lambda_{x-1}


    Args:
      hamiltonian (qml.Hamiltonian): The pennylane Hamiltonian to be decomposed

    Returns:
      dEs (List[float]): The energy (E_1-E-2)/2 separating the two eigenvalues of the spectrum
      mus (List[float]): The average between the two eigenvalues (E_1+E-2)/2
      times (List[float]): The time for this term group to be evaluated/evolved at
      projs (List[np.array]): The analytical observables associated with these groups,
       to be measured by qml.Hermitian
    """
    mat = hamiltonian.sparse_matrix().toarray()
    size = len(mat)
    eigs, eigvecs = np.linalg.eigh(mat)
    norm = eigs[-1]
    proj = np.identity(size, dtype='complex64')

    def Pi(j):
        """Projector on eigenspace of eigenvalue E_i"""
        return np.outer(np.conjugate(eigvecs[:, j]), eigvecs[:, j])
    proj += -2 * Pi(0)
    last_i = 1
    dEs, mus, projs, times = ([], [], [], [])
    for index in range(len(eigs) - 1):
        dE = (eigs[index + 1] - eigs[index]) / 2
        if np.isclose(dE, 0):
            continue
        dEs.append(dE)
        mu = (eigs[index + 1] + eigs[index]) / 2
        mus.append(mu)
        time = np.pi / (2 * (norm + abs(mu)))
        times.append(time)
        for j in range(last_i, index + 1):
            proj += -2 * Pi(j)
            last_i = index + 1
        projs.append(proj.copy() * dE)
    return (dEs, mus, times, projs)