from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest
def test_combo_index_shuffle():
    atoms, constr, bondcombo_def, target_bondcombo, anglecombo_def, target_anglecombo, dihedralcombo_def, target_dihedralcombo = setup_combos()
    answer = (0, 1, 2, 3, 4, 6, 7, 8)
    assert all((a == b for a, b in zip(constr.get_indices(), answer)))
    constr.index_shuffle(atoms, range(len(atoms)))
    assert all((a == b for a, b in zip(constr.get_indices(), answer)))
    constr.index_shuffle(atoms, [1, 2, 3, 4, 0, 7])
    assert constr.bondcombos[0][1] == [[1, 0, 1.0], [1, 2, -1.0]]
    assert constr.dihedralcombos[0][1] == [[2, 1, 0, 3, 1.0], [1, 0, 4, 5, 1.0]]