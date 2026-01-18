import numpy as np
from ase import Atom
from ase.build import bulk
from ase.calculators.checkpoint import Checkpoint, CheckpointCalculator
from ase.calculators.lj import LennardJones
from ase.lattice.cubic import Diamond
def rattle_calc(atoms, calc):
    orig_atoms = atoms.copy()
    np.random.seed(0)
    atoms.rattle()
    cp_calc_1 = CheckpointCalculator(calc)
    atoms.calc = cp_calc_1
    e11 = atoms.get_potential_energy()
    f11 = atoms.get_forces()
    atoms.rattle()
    e12 = atoms.get_potential_energy()
    f12 = atoms.get_forces()
    atoms = orig_atoms
    np.random.seed(0)
    atoms.rattle()
    cp_calc_2 = CheckpointCalculator(calc)
    atoms.calc = cp_calc_2
    e21 = atoms.get_potential_energy()
    f21 = atoms.get_forces()
    atoms.rattle()
    e22 = atoms.get_potential_energy()
    f22 = atoms.get_forces()
    assert e11 == e21
    assert e12 == e22
    assert np.abs(f11 - f21).max() < 1e-05
    assert np.abs(f12 - f22).max() < 1e-05