import os
import os.path
import numpy as np
import pytest
from ase import io
from ase.io import formats
from ase.build import bulk
@pytest.mark.parametrize('ext', compressions)
def test_modes(ext):
    """Test the different read/write modes for a compression format."""
    filename = 'testrw.{ext}'.format(ext=ext)
    for mode in ['w', 'wb', 'wt']:
        with formats.open_with_compression(filename, mode) as tmp:
            if 'b' in mode:
                tmp.write(b'some text')
            else:
                tmp.write('some text')
    for mode in ['r', 'rb', 'rt']:
        with formats.open_with_compression(filename, mode) as tmp:
            if 'b' in mode:
                assert tmp.read() == b'some text'
            else:
                assert tmp.read() == 'some text'