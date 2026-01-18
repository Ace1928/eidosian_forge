from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
def test_init_checks(self):
    mismatch_len = [self.h_atom.copy(), self.h2_molecule.copy()]
    with raises(ValueError, match='.*different numbers of atoms.*'):
        _ = neb.NEB(mismatch_len)
    mismatch_pbc = [self.h_atom.copy(), self.h_atom.copy()]
    mismatch_pbc[-1].set_pbc(True)
    with raises(ValueError, match='.*different boundary conditions.*'):
        _ = neb.NEB(mismatch_pbc)
    mismatch_numbers = [self.h_atom.copy(), Atoms('C', positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0])]
    with raises(ValueError, match='.*atoms in different orders.*'):
        _ = neb.NEB(mismatch_numbers)
    mismatch_cell = [self.h_atom.copy(), self.h_atom.copy()]
    mismatch_cell[-1].set_cell(mismatch_cell[-1].get_cell() + 1e-05)
    with raises(NotImplementedError, match='.*Variable cell.*'):
        _ = neb.NEB(mismatch_cell)