import os
from pathlib import Path
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule, bulk
import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.gui.gui import GUI
from ase.gui.save import save_dialog
from ase.gui.quickinfo import info
@pytest.mark.parametrize('filename', [None, 'output.png', 'output.eps', 'output.pov', 'output.traj', 'output.traj@0'])
def test_export_graphics(gui, testdir, with_bulk_ti, monkeypatch, filename):
    monkeypatch.setattr(ui.SaveFileDialog, 'go', lambda event: filename)
    gui.save()
    if filename is not None:
        realfilename = filename.rsplit('@')[0]
        assert Path(realfilename).is_file()