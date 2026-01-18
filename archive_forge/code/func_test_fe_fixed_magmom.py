import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.units import Hartree
def test_fe_fixed_magmom(fe_atoms):
    fe_atoms.calc.set(spinmagntarget=2.3)
    run(fe_atoms)