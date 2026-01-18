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
def test_nanotube(gui):
    nt = gui.nanotube_window()
    nt.apply()
    nt.element[1].value = '?'
    with pytest.raises(GUIError):
        nt.apply()
    nt.element[1].value = 'C'
    nt.ok()
    assert len(gui.images[0]) == 20