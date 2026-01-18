import os
import os.path
import numpy as np
import pytest
from ase import io
from ase.io import formats
from ase.build import bulk
@pytest.mark.parametrize('ext', compressions)
def test_compression_read_write_single(ext):
    """Re-reading a compressed file."""
    filename = 'single.xsf.{ext}'.format(ext=ext)
    io.write(filename, single)
    assert os.path.exists(filename)
    reread = io.read(filename)
    assert reread.get_chemical_symbols() == single.get_chemical_symbols()
    assert np.allclose(reread.positions, single.positions)