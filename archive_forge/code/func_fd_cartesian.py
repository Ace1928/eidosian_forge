import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
@pytest.fixture
def fd_cartesian():
    fd_cartesian = StringIO("\n    %chk=example.chk\n    %Nprocshared=16\n    # N B3LYP/6-31G(d',p') ! ASE formatted method and basis\n    # POpt(Tight, MaxCyc=100)/Integral=Ultrafine\n\n    Gaussian input prepared by ASE\n\n    0 1\n    8,  -0.464,   0.177,   0.0\n    1(iso=0.1134289259, NMagM=-8.89, ZEff=-1), -0.464,   1.137,   0.0\n    1(iso=2, spin=1, QMom=1, RadNuclear=1, ZNuc=2),   0.441,  -0.143,   0.0\n    TV        10.0000000000        0.0000000000        0.0000000000\n    TV         0.0000000000       10.0000000000        0.0000000000\n    TV         0.0000000000        0.0000000000       10.0000000000\n\n    ")
    return fd_cartesian