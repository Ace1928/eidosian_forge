import os
import os.path
import numpy as np
import pytest
from ase import io
from ase.io import formats
from ase.build import bulk
def test_get_compression():
    """Identification of supported compression from filename."""
    assert formats.get_compression('H2O.pdb.gz') == ('H2O.pdb', 'gz')
    assert formats.get_compression('CH4.pdb.bz2') == ('CH4.pdb', 'bz2')
    assert formats.get_compression('Alanine.pdb.xz') == ('Alanine.pdb', 'xz')
    assert formats.get_compression('DNA.pdb.zip') == ('DNA.pdb.zip', None)
    assert formats.get_compression('crystal.cif') == ('crystal.cif', None)