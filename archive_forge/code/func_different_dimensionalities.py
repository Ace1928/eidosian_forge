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
def different_dimensionalities():
    yield molecule('H2O')
    yield Atoms('X', cell=[1, 0, 0], pbc=[1, 0, 0])
    yield Atoms('X', cell=[1, 1, 0], pbc=[1, 1, 0])
    yield bulk('Au')