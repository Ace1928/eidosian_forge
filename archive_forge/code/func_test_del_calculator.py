import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
def test_del_calculator(atoms):
    atoms.calc = EMT()
    with pytest.deprecated_call():
        del atoms.calc
    assert atoms.calc is None