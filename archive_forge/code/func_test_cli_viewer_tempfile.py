import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
@pytest.mark.parametrize('viewer', CLIViewer.viewers())
def test_cli_viewer_tempfile(atoms, viewer):
    with viewer.mktemp(atoms) as path:
        atoms1 = read(path)
        assert all(atoms1.symbols == atoms.symbols)
    assert not path.exists()