from io import StringIO
import numpy as np
import pytest
from ase.io import read
from ase.io.formats import match_magic
import ase.units as units
def test_gaussian_out():
    fd = StringIO(buf)
    atoms = read(fd, format='gaussian-out')
    assert str(atoms.symbols) == 'OH2'
    assert atoms.positions == pytest.approx(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))
    assert not any(atoms.pbc)
    assert atoms.cell.rank == 0
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy / units.Ha == pytest.approx(-12.3456789)
    assert forces / (units.Ha / units.Bohr) == pytest.approx(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]))