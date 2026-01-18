from pathlib import Path
import numpy as np
import pytest
import ase.io
from ase.io import extxyz
from ase.atoms import Atoms
from ase.build import bulk
from ase.io.extxyz import escape
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCartesian
from ase.stress import full_3x3_to_voigt_6_stress
from ase.build import molecule
def test_json_scalars():
    a = bulk('Si')
    a.info['val_1'] = 42.0
    a.info['val_2'] = 42.0
    a.info['val_3'] = np.int64(42)
    a.write('tmp.xyz')
    with open('tmp.xyz', 'r') as fd:
        comment_line = fd.readlines()[1]
    assert 'val_1=42.0' in comment_line and 'val_2=42.0' in comment_line and ('val_3=42' in comment_line)
    b = ase.io.read('tmp.xyz')
    assert abs(b.info['val_1'] - 42.0) < 1e-06
    assert abs(b.info['val_2'] - 42.0) < 1e-06
    assert abs(b.info['val_3'] - 42) == 0