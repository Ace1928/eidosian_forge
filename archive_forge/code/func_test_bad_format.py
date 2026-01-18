from pathlib import Path
import pytest
from ase.io import read
from ase.io.formats import UnknownFileTypeError
def test_bad_format():
    path = mkfile('strangefile._no_such_format', 'strange file contents')
    with pytest.raises(UnknownFileTypeError, match='_no_such_format'):
        read(path)