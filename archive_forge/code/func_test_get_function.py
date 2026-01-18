import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_function(wan):
    nwannier = 2
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.0)
    nk = 2
    gpts = np.array([8, 8, 8])
    wanf = wan(atoms=atoms, gpts=gpts, kpts=(nk, nk, nk), rng=rng, nwannier=nwannier, initialwannier='bloch')
    assert (wanf.get_function(index=[0, 0]) == 0).all()
    assert wanf.get_function(index=[0, 1]) + wanf.get_function(index=[1, 0]) == pytest.approx(wanf.get_function(index=[1, 1]))
    for i in range(nwannier):
        assert (gpts * nk == wanf.get_function(index=i).shape).all()
        assert (gpts * [1, 2, 3] == wanf.get_function(index=i, repeat=[1, 2, 3]).shape).all()