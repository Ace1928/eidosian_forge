import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.units import Hartree
@pytest.mark.calculator_lite
def test_fe_any_magmom(fe_atoms):
    fe_atoms.calc.set(occopt=7)
    run(fe_atoms)