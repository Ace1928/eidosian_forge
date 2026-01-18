import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif_labels(cif_atoms):
    data = [['label' + str(i) for i in range(20)]]
    cif_atoms.write('testfile.cif', labels=data)
    atoms1 = read('testfile.cif', store_tags=True)
    print(atoms1.info)
    assert data[0] == atoms1.info['_atom_site_label']