from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def setup_fixinternals():
    atoms = setup_atoms()
    bond_def = [1, 2]
    target_bond = 1.4
    angle_def = [6, 0, 1]
    target_angle = atoms.get_angle(*angle_def)
    dihedral_def = [6, 0, 1, 2]
    target_dihedral = atoms.get_dihedral(*dihedral_def)
    constr = FixInternals(bonds=[(target_bond, bond_def)], angles_deg=[(target_angle, angle_def)], dihedrals_deg=[(target_dihedral, dihedral_def)], epsilon=1e-10)
    print(constr)
    return (atoms, constr, bond_def, target_bond, angle_def, target_angle, dihedral_def, target_dihedral)