import numpy as np
from ase.build import bulk
from ase.io.aims import read_aims as read
from ase.io.aims import parse_geometry_lines
from pytest import approx
def test_cartesian(atoms=atoms):
    """write cartesian coords and check if structure was preserved"""
    atoms.write(file, format=format)
    new_atoms = read(file)
    assert np.allclose(atoms.positions, new_atoms.positions)