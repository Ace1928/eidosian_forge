import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_hamiltonian_random(wan, rng):
    nwannier = 4
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.0)
    kpts = (2, 2, 2)
    Nk = kpts[0] * kpts[1] * kpts[2]
    wanf = wan(atoms=atoms, kpts=kpts, rng=rng, nwannier=nwannier, initialwannier='random')
    for k in range(Nk):
        H_ww = wanf.get_hamiltonian(k=k)
        for i in range(nwannier):
            for j in range(i + 1, nwannier):
                assert np.abs(H_ww[i, j]) == pytest.approx(np.abs(H_ww[j, i]))