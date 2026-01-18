import numpy as np
from numpy import linalg
from ase import units 
def get_dihedral_potential_gradient(atoms, dihedral):
    i = dihedral.atomi
    j = dihedral.atomj
    k = dihedral.atomk
    l = dihedral.atoml
    rij = rel_pos_pbc(atoms, i, j)
    rkj = rel_pos_pbc(atoms, k, j)
    dkj = linalg.norm(rkj)
    dkj2 = dkj * dkj
    rkl = rel_pos_pbc(atoms, k, l)
    rijrkj = np.dot(rij, rkj)
    rkjrkl = np.dot(rkj, rkl)
    rmj = np.cross(rij, rkj)
    dmj = linalg.norm(rmj)
    dmj2 = dmj * dmj
    emj = rmj / dmj
    rnk = np.cross(rkj, rkl)
    dnk = linalg.norm(rnk)
    dnk2 = dnk * dnk
    enk = rnk / dnk
    emjenk = np.dot(emj, enk)
    if np.abs(emjenk) > 1.0:
        emjenk = np.sign(emjenk)
    dddri = dkj / dmj2 * rmj
    dddrl = -dkj / dnk2 * rnk
    gx = np.zeros(12)
    gx[0:3] = dddri
    gx[3:6] = (rijrkj / dkj2 - 1.0) * dddri - rkjrkl / dkj2 * dddrl
    gx[6:9] = (rkjrkl / dkj2 - 1.0) * dddrl - rijrkj / dkj2 * dddri
    gx[9:12] = dddrl
    d = np.sign(np.dot(rkj, np.cross(rmj, rnk))) * np.arccos(emjenk)
    if dihedral.d0 is None:
        gx *= dihedral.k * np.sin(2.0 * d)
    else:
        dd = d - dihedral.d0
        dd = dd - np.around(dd / np.pi / 2.0) * np.pi * 2.0
        if dihedral.n is None:
            gx *= dihedral.k * dd
        else:
            gx *= -dihedral.k * dihedral.n * np.sin(dihedral.n * d - dihedral.d0)
    dihedral.d = d
    return (i, j, k, l, gx)