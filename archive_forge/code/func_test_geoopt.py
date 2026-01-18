from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
def test_geoopt(cp2k_factory, atoms):
    calc = cp2k_factory.calc(label='test_H2_GOPT', print_level='LOW')
    atoms.calc = calc
    with BFGS(atoms, logfile=None) as gopt:
        gopt.run(fmax=1e-06)
    dist = atoms.get_distance(0, 1)
    dist_ref = 0.7245595
    assert (dist - dist_ref) / dist_ref < 1e-07
    energy_ref = -30.7025616943
    energy = atoms.get_potential_energy()
    assert (energy - energy_ref) / energy_ref < 1e-10