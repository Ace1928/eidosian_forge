import pytest
import ase.build
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.geometry.dimensionality import (analyze_dimensionality,
def test_isolation_3D():
    atoms = FaceCenteredCubic(size=(2, 2, 2), symbol='Cu', pbc=(1, 1, 1))
    result = isolate_components(atoms)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '3D'
    assert len(components) == 1
    bulk = components[0]
    assert bulk.get_chemical_formula() == atoms.get_chemical_formula()