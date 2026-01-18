import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
def mock_view(self, atoms, repeat=None):
    print(f'viewing {atoms} with mock "{self.name}"')
    return (atoms, self.name)