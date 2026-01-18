import os
import os.path
import numpy as np
import pytest
from ase import io
from ase.io import formats
from ase.build import bulk
@pytest.mark.parametrize('ext', compressions)
def test_compression_write_multiple(ext):
    """Writing compressed file, with multiple configurations."""
    filename = 'multiple.xyz.{ext}'.format(ext=ext)
    io.write(filename, multiple)
    assert os.path.exists(filename)