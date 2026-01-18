import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_spectral_weight_random(wan, rng):
    nwannier = 4
    wanf = wan(initialwannier='random', nwannier=nwannier, rng=rng)
    for i in range(nwannier):
        assert wanf.get_spectral_weight(i).sum() == pytest.approx(1)