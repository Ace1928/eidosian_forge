from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
@pytest.mark.xfail
def test_combos():
    atoms, constr, bondcombo_def, target_bondcombo, anglecombo_def, target_anglecombo, dihedralcombo_def, target_dihedralcombo = setup_combos()
    ref_bondcombo = get_bondcombo(atoms, bondcombo_def)
    ref_anglecombo = get_anglecombo(atoms, anglecombo_def)
    ref_dihedralcombo = get_dihedralcombo(atoms, dihedralcombo_def)
    atoms.calc = EMT()
    atoms.set_constraint(constr)
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    new_bondcombo = get_bondcombo(atoms, bondcombo_def)
    new_anglecombo = get_anglecombo(atoms, anglecombo_def)
    new_dihedralcombo = get_dihedralcombo(atoms, dihedralcombo_def)
    err_bondcombo = new_bondcombo - ref_bondcombo
    err_anglecombo = new_anglecombo - ref_anglecombo
    err_dihedralcombo = new_dihedralcombo - ref_dihedralcombo
    print('error in bondcombo:', repr(err_bondcombo))
    print('error in anglecombo:', repr(err_anglecombo))
    print('error in dihedralcombo:', repr(err_dihedralcombo))
    for err in [err_bondcombo, err_anglecombo, err_dihedralcombo]:
        assert err < 1e-12