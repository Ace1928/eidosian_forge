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
def test_modify_tag(gui, modify):
    modify.tag.value = 17
    modify.set_tag()
    tags = gui.atoms.get_tags()
    assert all(tags[:4] == 17)
    assert all(tags[4:] == 0)