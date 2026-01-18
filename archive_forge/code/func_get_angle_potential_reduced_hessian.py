import numpy as np
from numpy import linalg
from ase import units 
def get_angle_potential_reduced_hessian(atoms, angle, morses=None):
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
    sina = np.sin(a)
    sina2 = sina * sina
    Pij = np.tensordot(eij, eij, axes=0)
    Qij = np.eye(3) - Pij
    Pkj = np.tensordot(ekj, ekj, axes=0)
    Qkj = np.eye(3) - Pkj
    Pki = np.tensordot(ekj, eij, axes=0)
    Hr = np.zeros((6, 6))
    if np.abs(sina) > 0.001:
        Hr[0:3, 0:3] = np.dot(Qij, np.dot(Pkj, Qij)) / dij2
        Hr[0:3, 3:6] = np.dot(Qij, np.dot(Pki, Qkj)) / dijdkj
        Hr[3:6, 0:3] = Hr[0:3, 3:6].T
        Hr[3:6, 3:6] = np.dot(Qkj, np.dot(Pij, Qkj)) / dkj2
    if angle.cos and np.abs(sina) > 0.001:
        cosa = np.cos(a)
        cosa0 = np.cos(angle.a0)
        factor = np.abs(1.0 - 2.0 * cosa * cosa + cosa * cosa0)
        Hr = Hr * factor * angle.k / sina2
    elif np.abs(sina) > 0.001:
        Hr = Hr * angle.k / sina2
    if angle.alpha is not None:
        Hr *= np.exp(angle.alpha[0] * (angle.rref[0] ** 2 - dij ** 2)) * np.exp(angle.alpha[1] * (angle.rref[1] ** 2 - dkj ** 2))
    if morses is not None:
        for m in range(len(morses)):
            if morses[m].atomi == i or morses[m].atomi == j or morses[m].atomi == k:
                Hr *= get_morse_potential_eta(atoms, morses[m])
            elif morses[m].atomj == i or morses[m].atomj == j or morses[m].atomj == k:
                Hr *= get_morse_potential_eta(atoms, morses[m])
    Hx = np.dot(Ax.T, np.dot(Hr, Ax))
    angle.a = a
    return (i, j, k, Hx)