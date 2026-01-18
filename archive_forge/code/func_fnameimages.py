import pytest
from ase.build import bulk, molecule
from ase.io import write
@pytest.fixture
def fnameimages(images, testdir):
    filename = 'fileimgs.xyz'
    write(filename, images)
    return filename