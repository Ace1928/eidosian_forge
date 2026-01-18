from pathlib import Path
import io
import pytest
from ase.build import molecule
from ase.io import read, write
from ase.utils import PurePath, convert_string_to_fd, reader, writer
@writer
def mywrite(file, fdcmp=None):
    assert isinstance(file, io.TextIOBase)
    assert file.mode == 'w'
    print(teststr, file=file)
    if fdcmp:
        assert file is fdcmp