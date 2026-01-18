import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_bad_occupancies(cif_atoms):
    assert 'Au' not in cif_atoms.symbols
    cif_atoms.symbols[0] = 'Au'
    with pytest.warns(UserWarning, match='no occupancy info'):
        write('tmp.cif', cif_atoms)