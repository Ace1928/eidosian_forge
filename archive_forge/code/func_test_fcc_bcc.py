from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_fcc_bcc(comparator):
    s1 = bulk('Al', crystalstructure='fcc')
    s2 = bulk('Al', crystalstructure='bcc', a=4.05)
    s1 = s1 * (2, 2, 2)
    s2 = s2 * (2, 2, 2)
    assert not comparator.compare(s1, s2)