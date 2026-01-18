import numpy as np
import pytest
from ase import Atom, Atoms
from ase.io.nwchem import write_nwchem_in
@pytest.fixture
def calculator_parameters():
    params = dict(memory='1024 mb', dft=dict(xc='b3lyp', mult=1, maxiter=300), basis='6-311G*')
    return params