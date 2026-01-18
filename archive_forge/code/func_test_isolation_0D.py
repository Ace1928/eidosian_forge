import pytest
import ase.build
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.geometry.dimensionality import (analyze_dimensionality,
@pytest.mark.parametrize('kcutoff', [None, 1.1])
def test_isolation_0D(kcutoff):
    atoms = ase.build.molecule('H2O', vacuum=3.0)
    result = isolate_components(atoms, kcutoff=kcutoff)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '0D'
    assert len(components) == 1
    molecule = components[0]
    assert molecule.get_chemical_formula() == atoms.get_chemical_formula()