import numpy as np
from numpy import linalg
from ase import units 
def get_dihedral_potential_value(atoms, dihedral):
    i = dihedral.atomi
    j = dihedral.atomj
    k = dihedral.atomk
    l = dihedral.atoml
    rij = rel_pos_pbc(atoms, i, j)
    rkj = rel_pos_pbc(atoms, k, j)
    rkl = rel_pos_pbc(atoms, k, l)
    rmj = np.cross(rij, rkj)
    dmj = linalg.norm(rmj)
    emj = rmj / dmj
    rnk = np.cross(rkj, rkl)
    dnk = linalg.norm(rnk)
    enk = rnk / dnk
    emjenk = np.dot(emj, enk)
    if np.abs(emjenk) > 1.0:
        emjenk = np.sign(emjenk)
    d = np.sign(np.dot(rkj, np.cross(rmj, rnk))) * np.arccos(emjenk)
    if dihedral.d0 is None:
        v = 0.5 * dihedral.k * (1.0 - np.cos(2.0 * d))
    else:
        dd = d - dihedral.d0
        dd = dd - np.around(dd / np.pi / 2.0) * np.pi * 2.0
        if dihedral.n is None:
            v = 0.5 * dihedral.k * dd ** 2
        else:
            v = dihedral.k * (1.0 + np.cos(dihedral.n * d - dihedral.d0))
    dihedral.d = d
    return (i, j, k, l, v)