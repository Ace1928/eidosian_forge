import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_empty_or_atomless_cifblock():
    ciffile = io.BytesIO(b'data_dummy')
    blocks = list(parse_cif(ciffile))
    assert len(blocks) == 1
    assert not blocks[0].has_structure()
    with pytest.raises(NoStructureData):
        blocks[0].get_atoms()