from pytest import warns, raises
from ase import Atoms
from ase import neb
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
def test_neb_default(self):
    neb_dummy = neb.NEB(self.images_dummy)
    assert not neb_dummy.allow_shared_calculator