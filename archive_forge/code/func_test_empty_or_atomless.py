import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
@pytest.mark.parametrize('data', [b'', b'data_dummy'])
def test_empty_or_atomless(data):
    ciffile = io.BytesIO(data)
    images = read(ciffile, index=':', format='cif')
    assert len(images) == 0