import pytest
import ase.build
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.geometry.dimensionality import (analyze_dimensionality,
def test_isolation_1D():
    atoms = Atoms(symbols='Cl6Ti2', pbc=True, cell=[[6.27, 0, 0], [-3.135, 5.43, 0], [0, 0, 5.82]], positions=[[1.97505, 0, 1.455], [0.987525, 1.71044347, 4.365], [-0.987525, 1.71044347, 1.455], [4.29495, 0, 4.365], [2.147475, 3.71953581, 1.455], [-2.147475, 3.71953581, 4.365], [0, 0, 0], [0, 0, 2.91]])
    result = isolate_components(atoms)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '1D'
    assert len(components) == 1
    chain = components[0]
    assert (chain.pbc == [False, False, True]).all()
    assert chain.get_chemical_formula() == atoms.get_chemical_formula()