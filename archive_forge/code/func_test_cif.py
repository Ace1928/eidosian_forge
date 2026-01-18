import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif():
    cif_file = io.StringIO(content)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        atoms_leg = read(cif_file, format='cif', fractional_occupancies=False)
    elements = np.unique(atoms_leg.get_atomic_numbers())
    for n in (11, 17, 53):
        assert n in elements
    try:
        atoms_leg.info['occupancy']
        raise AssertionError
    except KeyError:
        pass
    cif_file = io.StringIO(content)
    atoms = read(cif_file, format='cif', fractional_occupancies=True)
    assert len(atoms_leg) == len(atoms)
    assert np.all(atoms_leg.get_atomic_numbers() == atoms.get_atomic_numbers())
    assert atoms_leg == atoms
    elements = np.unique(atoms_leg.get_atomic_numbers())
    for n in (11, 17, 53):
        assert n in elements
    check_fractional_occupancies(atoms)
    fname = 'testfile.cif'
    with open(fname, 'wb') as fd:
        write(fd, atoms, format='cif')
    with open(fname) as fd:
        atoms = read(fd, format='cif', fractional_occupancies=True)
    check_fractional_occupancies(atoms)
    atoms = atoms.repeat([2, 1, 1])
    assert len(atoms.arrays['spacegroup_kinds']) == len(atoms.arrays['numbers'])