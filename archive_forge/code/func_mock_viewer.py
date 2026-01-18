import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
@pytest.fixture
def mock_viewer():
    return CLIViewer('dummy', 'traj', [sys.executable, '-m', 'ase', 'info'])