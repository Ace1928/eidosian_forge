from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.io import read
from ase.constraints import FixAtoms
def check_forces():
    """Makes sure the unconstrained forces stay that way."""
    forces = atoms.get_forces(apply_constraint=False)
    funconstrained = float(forces[0, 0])
    forces = atoms.get_forces(apply_constraint=True)
    forces = atoms.get_forces(apply_constraint=False)
    funconstrained2 = float(forces[0, 0])
    assert funconstrained2 == funconstrained