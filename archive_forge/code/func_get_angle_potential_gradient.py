import numpy as np
from numpy import linalg
from ase import units 
def get_angle_potential_gradient(atoms, angle):
    i = angle.atomi
    j = angle.atomj
    k = angle.atomk
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    rkj = rel_pos_pbc(atoms, k, j)
    dkj = linalg.norm(rkj)
    ekj = rkj / dkj
    eijekj = np.dot(eij, ekj)
    if np.abs(eijekj) > 1.0:
        eijekj = np.sign(eijekj)
    a = np.arccos(eijekj)
    if angle.cos:
        da = np.cos(a) - np.cos(angle.a0)
    else:
        da = a - angle.a0
        da = da - np.around(da / np.pi) * np.pi
        sina = np.sin(a)
    Pij = np.tensordot(eij, eij, axes=0)
    Qij = np.eye(3) - Pij
    Pkj = np.tensordot(ekj, ekj, axes=0)
    Qkj = np.eye(3) - Pkj
    gr = np.zeros(6)
    if angle.cos:
        gr[0:3] = angle.k * da / dij * np.dot(Qij, ekj)
        gr[3:6] = angle.k * da / dkj * np.dot(Qkj, eij)
    elif np.abs(sina) > 0.001:
        gr[0:3] = -angle.k * da / sina / dij * np.dot(Qij, ekj)
        gr[3:6] = -angle.k * da / sina / dkj * np.dot(Qkj, eij)
    gx = np.dot(Ax.T, gr)
    angle.a = a
    return (i, j, k, gx)