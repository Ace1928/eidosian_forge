import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def test_export_xyz(at0, testdir):
    """
    test the export_extxyz function and checks the region adn forces arrays
    """
    forces = at0.get_forces()
    filename = 'qmmm_export_test.xyz'
    qmmm = at0.calc
    qmmm.export_extxyz(filename=filename)
    from ase.io import read
    read_atoms = read(filename)
    assert 'region' in read_atoms.arrays
    original_region = qmmm.get_region_from_masks()
    assert all(original_region == read_atoms.get_array('region'))
    assert 'forces' in read_atoms.arrays
    np.testing.assert_allclose(forces, read_atoms.get_forces(), atol=1e-06)