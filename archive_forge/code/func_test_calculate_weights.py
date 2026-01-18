import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
@pytest.mark.parametrize('lat', bravais_lattices())
def test_calculate_weights(lat):
    tol = 1e-05
    cell = lat.tocell()
    g = cell @ cell.T
    w, G = calculate_weights(cell, normalize=False)
    errors = []
    for i in range(3):
        for j in range(3):
            errors.append(np.abs(w * G[:, i] @ G[:, j] - g[i, j]))
    assert np.max(errors) < tol