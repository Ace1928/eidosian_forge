import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif_loop_keys(cif_atoms):
    data = {}
    data['someKey'] = [[str(i) + 'test' for i in range(20)]]
    data['someIntKey'] = [[str(i) + '123' for i in range(20)]]
    cif_atoms.write('testfile.cif', loop_keys=data)
    atoms1 = read('testfile.cif', store_tags=True)
    r_data = {'someKey': atoms1.info['_somekey'], 'someIntKey': atoms1.info['_someintkey']}
    assert r_data['someKey'] == data['someKey'][0]
    assert r_data['someIntKey'] == [int(x) for x in data['someIntKey'][0]]