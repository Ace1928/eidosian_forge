import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
def test_cli_viewer_blocking(atoms, mock_viewer):
    mock_viewer.view_blocking(atoms)