import numpy as np
from numpy import linalg
from ase import units 
def get_vdw_potential_hessian(atoms, vdw, spectral=False):
    i = vdw.atomi
    j = vdw.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    Pij = np.tensordot(eij, eij, axes=0)
    Qij = np.eye(3) - Pij
    Hr = (156.0 * vdw.Aij / dij ** 14 - 42.0 * vdw.Bij / dij ** 8) * Pij + (-12.0 * vdw.Aij / dij ** 13 + 6.0 * vdw.Bij / dij ** 7) / dij * Qij
    Hx = np.dot(Bx.T, np.dot(Hr, Bx))
    if spectral:
        eigvals, eigvecs = linalg.eigh(Hx)
        D = np.diag(np.abs(eigvals))
        U = eigvecs
        Hx = np.dot(U, np.dot(D, np.transpose(U)))
    vdw.r = dij
    return (i, j, Hx)