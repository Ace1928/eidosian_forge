import pytest
from ase.build import bulk, molecule
from ase.io import write
@pytest.fixture
def execfilename(testdir):
    filename = 'execcode.py'
    with open(filename, 'w') as fd:
        fd.write('print(len(atoms))')
    return filename