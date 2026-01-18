import io
from ase.io import read
def test_gpaw_output():
    """Regression test for #896.

    "ase.io does not read all configurations from gpaw-out file"

    """
    fd = io.StringIO(text)
    configs = read(fd, index=':', format='gpaw-out')
    assert len(configs) == 3