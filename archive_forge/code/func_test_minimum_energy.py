import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
def test_minimum_energy():
    for atoms in systems_minimum():
        assert atoms.get_potential_energy() == reference_potential_energy
        assert atoms.get_potential_energies().sum() == reference_potential_energy