import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
def test_get_calculator(atoms):
    with pytest.deprecated_call():
        assert atoms.get_calculator() is None