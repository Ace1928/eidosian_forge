import numpy as np
from numpy import linalg
from ase import units 
def get_vdw_potential_value(atoms, vdw):
    i = vdw.atomi
    j = vdw.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    v = vdw.Aij / dij ** 12 - vdw.Bij / dij ** 6
    vdw.r = dij
    return (i, j, v)