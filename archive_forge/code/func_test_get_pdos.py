import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_get_pdos(wan):
    nwannier = 4
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(gpts=(16, 16, 16), nbands=nwannier, txt=None)
    atoms = molecule('H2')
    atoms.center(vacuum=3.0)
    atoms.calc = calc
    atoms.get_potential_energy()
    wanf = wan(atoms=atoms, calc=calc, nwannier=nwannier, initialwannier='bloch')
    eig_n = calc.get_eigenvalues()
    for i in range(nwannier):
        pdos_n = wanf.get_pdos(w=i, energies=eig_n, width=0.001)
        assert pdos_n[i] != pytest.approx(0)