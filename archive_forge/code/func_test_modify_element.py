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
def test_modify_element(gui, modify):

    class MockElement:
        Z = 79
    modify.set_element(MockElement())
    assert all(gui.atoms.symbols[:4] == 'Au')
    assert all(gui.atoms.symbols[4:] == 'Ti')