import numpy as np
import pytest
from ase.io.opls import OPLSff, OPLSStructure
@pytest.fixture
def opls_structure_file_name(datadir):
    return datadir / 'opls_structure_ext.xyz'