import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cifloop():
    dct = {'_eggs': range(4), '_potatoes': [1.3, 7.1, -1, 0]}
    loop = CIFLoop()
    loop.add('_eggs', dct['_eggs'], '{:<2d}')
    loop.add('_potatoes', dct['_potatoes'], '{:.4f}')
    string = loop.tostring() + '\n\n'
    lines = string.splitlines()[::-1]
    assert lines.pop() == 'loop_'
    newdct = parse_loop(lines)
    print(newdct)
    assert set(dct) == set(newdct)
    for name in dct:
        assert dct[name] == pytest.approx(newdct[name])