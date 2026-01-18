import numpy as np
import pytest
from ase import Atoms
from ase.io import qbox
from ase.io import formats
@pytest.fixture
def qballfile(datadir):
    return datadir / 'qbox_04_md_ntc.reference.xml'