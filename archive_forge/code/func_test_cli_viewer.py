import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
def test_cli_viewer(atoms, mock_viewer):
    handle = mock_viewer.view(atoms)
    status = handle.wait()
    assert status == 0