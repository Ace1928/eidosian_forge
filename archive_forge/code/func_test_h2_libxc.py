from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
def test_h2_libxc(cp2k_factory, atoms):
    calc = cp2k_factory.calc(xc='XC_GGA_X_PBE XC_GGA_C_PBE', pseudo_potential='GTH-PBE', label='test_H2_libxc')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -31.591716529642
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10