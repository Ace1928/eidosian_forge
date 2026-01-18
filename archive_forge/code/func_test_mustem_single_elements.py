import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.build import bulk
from ase.atoms import symbols2numbers
def test_mustem_single_elements():
    Si_atoms = bulk('Si', cubic=True)
    filename = 'Si100.xtl'
    DW = 0.62
    Si_atoms.write(filename, keV=300, debye_waller_factors=DW)
    Si_atoms2 = read(filename)
    np.testing.assert_allclose(Si_atoms.positions, Si_atoms2.positions)
    np.testing.assert_allclose(Si_atoms.cell, Si_atoms2.cell)
    np.testing.assert_allclose(Si_atoms2.arrays['occupancies'], np.ones(8))
    np.testing.assert_allclose(Si_atoms2.arrays['debye_waller_factors'], np.ones(8) * DW, rtol=0.01)
    Si_atoms3 = bulk('Si', cubic=True)
    Si_atoms3.set_array('occupancies', np.ones(8) * 0.9)
    Si_atoms3.set_array('debye_waller_factors', np.ones(8) * DW)
    Si_atoms3.write(filename, keV=300)
    Si_atoms4 = read(filename)
    np.testing.assert_allclose(Si_atoms3.positions, Si_atoms4.positions)
    np.testing.assert_allclose(Si_atoms3.cell, Si_atoms4.cell)
    np.testing.assert_allclose(Si_atoms3.arrays['occupancies'], Si_atoms4.arrays['occupancies'])
    np.testing.assert_allclose(Si_atoms3.arrays['debye_waller_factors'], Si_atoms4.arrays['debye_waller_factors'], rtol=0.01)
    Si_atoms5 = bulk('Si', cubic=True)
    debye_waller_factors = np.ones(8) * DW
    debye_waller_factors[0] = debye_waller_factors[0] / 2
    Si_atoms5.set_array('debye_waller_factors', debye_waller_factors)
    with pytest.raises(ValueError):
        Si_atoms5.write(filename, keV=300)