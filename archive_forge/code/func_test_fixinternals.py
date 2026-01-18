from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def test_fixinternals():
    atoms, constr, bond_def, target_bond, angle_def, target_angle, dihedral_def, target_dihedral = setup_fixinternals()
    calc = EMT()
    opt = BFGS(atoms)
    previous_angle = atoms.get_angle(*angle_def)
    previous_dihedral = atoms.get_dihedral(*dihedral_def)
    print('angle before', previous_angle)
    print('dihedral before', previous_dihedral)
    print('bond length before', atoms.get_distance(*bond_def))
    print('target bondlength', target_bond)
    atoms.calc = calc
    atoms.set_constraint(constr)
    print('-----Optimization-----')
    opt.run(fmax=0.01)
    new_angle = atoms.get_angle(*angle_def)
    new_dihedral = atoms.get_dihedral(*dihedral_def)
    new_bondlength = atoms.get_distance(*bond_def)
    print('angle after', new_angle)
    print('dihedral after', new_dihedral)
    print('bondlength after', new_bondlength)
    err1 = new_angle - previous_angle
    err2 = new_dihedral - previous_dihedral
    err3 = new_bondlength - target_bond
    print('error in angle', repr(err1))
    print('error in dihedral', repr(err2))
    print('error in bondlength', repr(err3))
    assert err1 < 1e-11
    assert err2 < 1e-12
    assert err3 < 1e-12