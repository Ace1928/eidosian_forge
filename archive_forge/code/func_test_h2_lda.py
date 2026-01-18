from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
def test_h2_lda(cp2k_factory, atoms):
    calc = cp2k_factory.calc(label='test_H2_LDA')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -30.6989595886
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10