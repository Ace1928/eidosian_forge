import numpy as np
import pytest
from ase.build import bulk
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.harmonic import SpringCalculator
from ase.md.switch_langevin import SwitchLangevin
@pytest.mark.slow
def test_langevin_switching():
    size = 6
    T = 300
    n_steps = 500
    k1 = 2.0
    k2 = 4.0
    dt = 10
    np.random.seed(42)
    atoms = bulk('Al').repeat(size)
    calc1 = SpringCalculator(atoms.positions, k1)
    calc2 = SpringCalculator(atoms.positions, k2)
    n_atoms = len(atoms)
    calc1.atoms = atoms
    calc2.atoms = atoms
    F1 = calc1.get_free_energy(T) / n_atoms
    F2 = calc2.get_free_energy(T) / n_atoms
    dF_theory = F2 - F1
    with SwitchLangevin(atoms, calc1, calc2, dt * units.fs, temperature_K=T, friction=0.01, n_eq=n_steps, n_switch=n_steps) as dyn_forward:
        MaxwellBoltzmannDistribution(atoms, temperature_K=2 * T)
        dyn_forward.run()
        dF_forward = dyn_forward.get_free_energy_difference() / len(atoms)
    with SwitchLangevin(atoms, calc2, calc1, dt * units.fs, temperature_K=T, friction=0.01, n_eq=n_steps, n_switch=n_steps) as dyn_backward:
        MaxwellBoltzmannDistribution(atoms, temperature_K=2 * T)
        dyn_backward.run()
        dF_backward = -dyn_backward.get_free_energy_difference() / len(atoms)
    dF_switch = (dF_forward + dF_backward) / 2.0
    error = dF_switch - dF_theory
    assert abs(error) < 0.001