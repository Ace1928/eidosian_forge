import numpy as np
from numpy import linalg
from ase import units 
def get_angle_potential_hessian(atoms, angle, morses=None, spectral=False):
    i = angle.atomi
    j = angle.atomj
    k = angle.atomk
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    dij2 = dij * dij
    eij = rij / dij
    rkj = rel_pos_pbc(atoms, k, j)
    dkj = linalg.norm(rkj)
    dkj2 = dkj * dkj
    ekj = rkj / dkj
    dijdkj = dij * dkj
    eijekj = np.dot(eij, ekj)
    if np.abs(eijekj) > 1.0:
        eijekj = np.sign(eijekj)
    a = np.arccos(eijekj)
    if angle.cos:
        da = np.cos(a) - np.cos(angle.a0)
        cosa0 = np.cos(angle.a0)
    else:
        da = a - angle.a0
        da = da - np.around(da / np.pi) * np.pi
    sina = np.sin(a)
    cosa = np.cos(a)
    ctga = cosa / sina
    Pij = np.tensordot(eij, eij, axes=0)
    Qij = np.eye(3) - Pij
    Pkj = np.tensordot(ekj, ekj, axes=0)
    Qkj = np.eye(3) - Pkj
    Pik = np.tensordot(eij, ekj, axes=0)
    Pki = np.tensordot(ekj, eij, axes=0)
    P = np.eye(3) * eijekj
    QijPkjQij = np.dot(Qij, np.dot(Pkj, Qij))
    QijPkiQkj = np.dot(Qij, np.dot(Pki, Qkj))
    QkjPijQkj = np.dot(Qkj, np.dot(Pij, Qkj))
    Hr = np.zeros((6, 6))
    if angle.cos and np.abs(sina) > 0.001:
        factor = 1.0 - 2.0 * cosa * cosa + cosa * cosa0
        Hr[0:3, 0:3] = angle.k * (factor * QijPkjQij / sina - sina * da * (-ctga * QijPkjQij / sina + np.dot(Qij, Pki) - np.dot(Pij, Pki) * 2.0 + (Pik + P))) / sina / dij2
        Hr[0:3, 3:6] = angle.k * (factor * QijPkiQkj / sina - sina * da * (-ctga * QijPkiQkj / sina - np.dot(Qij, Qkj))) / sina / dijdkj
        Hr[3:6, 0:3] = Hr[0:3, 3:6].T
        Hr[3:6, 3:6] = angle.k * (factor * QkjPijQkj / sina - sina * da * (-ctga * QkjPijQkj / sina + np.dot(Qkj, Pik) - np.dot(Pkj, Pik) * 2.0 + (Pki + P))) / sina / dkj2
    elif np.abs(sina) > 0.001:
        Hr[0:3, 0:3] = angle.k * (QijPkjQij / sina + da * (-ctga * QijPkjQij / sina + np.dot(Qij, Pki) - np.dot(Pij, Pki) * 2.0 + (Pik + P))) / sina / dij2
        Hr[0:3, 3:6] = angle.k * (QijPkiQkj / sina + da * (-ctga * QijPkiQkj / sina - np.dot(Qij, Qkj))) / sina / dijdkj
        Hr[3:6, 0:3] = Hr[0:3, 3:6].T
        Hr[3:6, 3:6] = angle.k * (QkjPijQkj / sina + da * (-ctga * QkjPijQkj / sina + np.dot(Qkj, Pik) - np.dot(Pkj, Pik) * 2.0 + (Pki + P))) / sina / dkj2
    if angle.alpha is not None:
        Hr *= np.exp(angle.alpha[0] * (angle.rref[0] ** 2 - dij ** 2)) * np.exp(angle.alpha[1] * (angle.rref[1] ** 2 - dkj ** 2))
    if morses is not None:
        for m in range(len(morses)):
            if morses[m].atomi == i or morses[m].atomi == j or morses[m].atomi == k:
                Hr *= get_morse_potential_eta(atoms, morses[m])
            elif morses[m].atomj == i or morses[m].atomj == j or morses[m].atomj == k:
                Hr *= get_morse_potential_eta(atoms, morses[m])
    Hx = np.dot(Ax.T, np.dot(Hr, Ax))
    if spectral:
        eigvals, eigvecs = linalg.eigh(Hx)
        D = np.diag(np.abs(eigvals))
        U = eigvecs
        Hx = np.dot(U, np.dot(D, np.transpose(U)))
    angle.a = a
    return (i, j, k, Hx)