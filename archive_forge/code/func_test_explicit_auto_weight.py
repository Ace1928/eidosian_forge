import os
import pytest
from ase.build import bulk
from .filecmp_ignore_whitespace import filecmp_ignore_whitespace
@calc('vasp')
def test_explicit_auto_weight(factory, write_kpoints):
    write_kpoints(factory, kpts=[(0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.0, 0.5, 0.5)], reciprocal=True)
    with open('KPOINTS.ref', 'w') as fd:
        fd.write('KPOINTS created by Atomic Simulation Environment\n    3 \n    Reciprocal\n    0.100000 0.200000 0.300000 1.0 \n    0.000000 0.000000 0.000000 1.0 \n    0.000000 0.500000 0.500000 1.0 \n    ')
    assert filecmp_ignore_whitespace('KPOINTS', 'KPOINTS.ref')