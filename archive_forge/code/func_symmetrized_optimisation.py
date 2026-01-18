import numpy as np
import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.lj import LennardJones
from ase.spacegroup.symmetrize import FixSymmetry, check_symmetry, is_subgroup
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.constraints import UnitCellFilter, ExpCellFilter
def symmetrized_optimisation(at_init, filter):
    rng = np.random.RandomState(1)
    at = at_init.copy()
    at.calc = NoisyLennardJones(rng=rng)
    at_cell = filter(at)
    print('Initial Energy', at.get_potential_energy(), at.get_volume())
    with PreconLBFGS(at_cell, precon=None) as dyn:
        dyn.run(steps=300, fmax=0.001)
        print('n_steps', dyn.get_number_of_steps())
    print('Final Energy', at.get_potential_energy(), at.get_volume())
    print('Final forces\n', at.get_forces())
    print('Final stress\n', at.get_stress())
    print('initial symmetry at 1e-6')
    di = check_symmetry(at_init, 1e-06, verbose=True)
    print('final symmetry at 1e-6')
    df = check_symmetry(at, 1e-06, verbose=True)
    return (di, df)