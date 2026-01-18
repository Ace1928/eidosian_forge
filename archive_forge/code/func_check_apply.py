import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms, UnitCellFilter
from ase.calculators.emt import EMT
from ase.optimize.precon import make_precon, Precon
from ase.neighborlist import neighbor_list
from ase.utils.ff import Bond
def check_apply(precon, system):
    atoms, bonds = system
    kwargs = {}
    if precon == 'FF' or precon == 'Exp_FF':
        kwargs['bonds'] = bonds
    precon = make_precon(precon, atoms, **kwargs)
    forces = atoms.get_forces().reshape(-1)
    precon_forces, residual = precon.apply(forces, atoms)
    residual_P = np.linalg.norm(precon_forces, np.inf)
    print(f'|F| = {residual:.3f} |F|_P = {np.linalg.norm(precon_forces, np.inf):.3f}')
    assert residual_P <= residual
    fixed_atoms = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.extend(list(constraint.index))
    if len(fixed_atoms) != 0:
        assert np.linalg.norm(forces[fixed_atoms], np.inf) < 1e-08
        assert np.linalg.norm(precon_forces[fixed_atoms], np.inf) < 1e-08