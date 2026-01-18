from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_mirror_plane(comparator):
    s1 = get_atoms_with_mixed_elements(crystalstructure='hcp')
    s2 = s1.copy()
    mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    s2.set_positions(mat.dot(s2.get_positions().T).T)
    assert comparator.compare(s1, s2)
    mat = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    s2.set_positions(mat.dot(s1.get_positions().T).T)
    assert comparator.compare(s1, s2)
    mat = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    s2.set_positions(mat.dot(s1.get_positions().T).T)
    assert comparator.compare(s1, s2)