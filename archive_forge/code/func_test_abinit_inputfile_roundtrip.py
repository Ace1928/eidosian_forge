from io import StringIO
import numpy as np
import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.abinit import read_abinit_out, read_eig, match_kpt_header
from ase.units import Hartree, Bohr
def test_abinit_inputfile_roundtrip(testdir):
    m1 = bulk('Ti')
    m1.set_initial_magnetic_moments(range(len(m1)))
    write('abinit_save.in', images=m1, format='abinit-in')
    m2 = read('abinit_save.in', format='abinit-in')
    assert not compare_atoms(m1, m2, tol=1e-07)