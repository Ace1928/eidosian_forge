import numpy as np
import pytest
from ase.build import molecule
from ase.utils.ff import Morse, Angle, Dihedral, VdW
from ase.calculators.ff import ForceField
from ase.optimize.precon.neighbors import get_neighbours
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon import FF
@pytest.fixture(scope='module')
def forcefield_params(atoms0):
    a = atoms0
    cutoff = 1.5
    morse_D = 6.1322
    morse_alpha = 1.8502
    morse_r0 = 1.4322
    angle_k = 10.0
    angle_a0 = np.deg2rad(120.0)
    dihedral_k = 0.346
    vdw_epsilonij = 0.0115
    vdw_rminij = 3.4681
    neighbor_list = [[] for _ in range(len(a))]
    vdw_list = np.ones((len(a), len(a)), dtype=bool)
    morses = []
    angles = []
    dihedrals = []
    vdws = []
    i_list, j_list, d_list, fixed_atoms = get_neighbours(atoms=a, r_cut=cutoff)
    for i, j in zip(i_list, j_list):
        neighbor_list[i].append(j)
    for i in range(len(neighbor_list)):
        neighbor_list[i].sort()
    for i in range(len(a)):
        for jj in range(len(neighbor_list[i])):
            j = neighbor_list[i][jj]
            if j > i:
                morses.append(Morse(atomi=i, atomj=j, D=morse_D, alpha=morse_alpha, r0=morse_r0))
            vdw_list[i, j] = vdw_list[j, i] = False
            for kk in range(jj + 1, len(neighbor_list[i])):
                k = neighbor_list[i][kk]
                angles.append(Angle(atomi=j, atomj=i, atomk=k, k=angle_k, a0=angle_a0, cos=True))
                vdw_list[j, k] = vdw_list[k, j] = False
                for ll in range(kk + 1, len(neighbor_list[i])):
                    l = neighbor_list[i][ll]
                    dihedrals.append(Dihedral(atomi=j, atomj=i, atomk=k, atoml=l, k=dihedral_k))
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if vdw_list[i, j]:
                vdws.append(VdW(atomi=i, atomj=j, epsilonij=vdw_epsilonij, rminij=vdw_rminij))
    return dict(morses=morses, angles=angles, dihedrals=dihedrals, vdws=vdws)